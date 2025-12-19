"""
covid_engine.py
Raw(TXT + grades Excel) -> processed tables -> analysis results in one run.

Usage example:
  python covid_engine.py \
    --raw_txt_dir data/raw \
    --grades_excel "data/raw/학생_성적_정보_양식_통합.xlsx" \
    --out_dir out \
    --group_mode cohort

Outputs:
  out/processed/student_info.csv
  out/processed/grades.csv
  out/processed/seteuk.csv
  out/processed/volatility.csv
  out/results/group_compare_summary.csv
  out/results/ols_results.csv
  out/results/mediation_results.csv  (if enabled)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# statsmodels is used for OLS + robust (HC3) standard errors
import statsmodels.api as sm


# ---------------------------------------------------------------------
# 1) Reuse the proven TXT parser / grades loader from your STEP1
#    (We are NOT "running step1.py" as a script. We import its parsing API.)
# ---------------------------------------------------------------------
try:
    import step1_parse_all_files as step1
except Exception as e:
    raise RuntimeError(
        "step1_parse_all_files.py 모듈을 import할 수 없습니다.\n"
        "covid_engine.py와 step1_parse_all_files.py를 같은 폴더에 두거나, "
        "PYTHONPATH에 포함되도록 실행해 주세요.\n"
        f"원인: {e}"
    ) from e


# ---------------------------------------------------------------------
# 2) Config
# ---------------------------------------------------------------------
@dataclass
class EngineConfig:
    raw_txt_dir: Path
    grades_excel: Path
    out_dir: Path
    group_mode: str = "cohort"  # cohort | any_covid | grade3_covid | remote_days_threshold | repeat
    remote_days_threshold: int = 1
    run_mediation: bool = False
    mediation_bootstrap: int = 1000
    random_seed: int = 42


# ---------------------------------------------------------------------
# 3) Ingest + Transform
# ---------------------------------------------------------------------
def parse_txt_files(raw_txt_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all TXT files in raw_txt_dir into:
      - df_students (one row per student file)
      - df_seteuk (many rows per student: grade/subject/text/keyword counts, etc.)
    """
    raw_txt_dir = Path(raw_txt_dir)
    txt_files = sorted(raw_txt_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"TXT 파일을 찾을 수 없습니다: {raw_txt_dir}")

    parser = step1.FinalParser()

    students: List[Dict] = []
    seteuk_rows: List[Dict] = []

    for fp in txt_files:
        try:
            text = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = fp.read_text(encoding="cp949", errors="ignore")

        info = parser.parse_student_info(text, fp.name)
        students.append(info)

        grade_years = info.get("grade_years", {1: None, 2: None, 3: None})
        sid = info.get("student_id", "")

        rows = parser.extract_seteuk(text, student_id=sid, grade_years=grade_years)
        if rows:
            seteuk_rows.extend(rows)

    df_students = pd.DataFrame(students)
    df_seteuk = pd.DataFrame(seteuk_rows)

    # JSON-like dict columns (remote_days, grade_years) are kept; we also provide serialized strings for CSV safety.
    for col in ["remote_days", "grade_years"]:
        if col in df_students.columns:
            df_students[f"{col}_json"] = df_students[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x))

    return df_students, df_seteuk


def load_grades(grades_excel: Path) -> pd.DataFrame:
    """Load grades from the standardized Excel."""
    grades_excel = Path(grades_excel)
    if not grades_excel.exists():
        raise FileNotFoundError(f"성적 엑셀 파일이 없습니다: {grades_excel}")
    return step1.load_grades_from_excel(str(grades_excel))


def calculate_volatility(df_students: pd.DataFrame, df_grades: pd.DataFrame) -> pd.DataFrame:
    """Compute per-student volatility metrics using STEP1's stable implementation."""
    rows: List[Dict] = []
    for _, r in df_students.iterrows():
        sid = str(r.get("student_id", ""))
        remote_days = r.get("remote_days", {})
        if isinstance(remote_days, str):
            # try to parse dict string if needed
            try:
                remote_days = json.loads(remote_days)
            except Exception:
                remote_days = {}
        rows.append(step1.calculate_volatility_from_df(df_grades, sid, remote_days if isinstance(remote_days, dict) else {}))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 4) Grouping (Corona student group definitions)
# ---------------------------------------------------------------------
def _parse_remote_days_sum(v) -> int:
    if isinstance(v, dict):
        return int(sum(v.values()))
    if isinstance(v, str):
        # rough parse: "{1: 10, 2: 0, 3: 5}" -> 10+0+5
        nums = [int(x) for x in re.findall(r"\d+", v)]
        if len(nums) >= 2:
            return int(sum(nums[1::2]))
    return 0


def build_group_label(df_students: pd.DataFrame, mode: str, remote_days_threshold: int = 1) -> pd.Series:
    """
    Return 0/1 group label.
      - cohort: Pre-COVID vs COVID based on cohort column (hs_graduation_year >= 2021 => COVID)
      - any_covid: any_covid 0/1
      - grade3_covid: grade3_covid 0/1
      - remote_days_threshold: sum(remote_days) >= threshold
      - repeat: is_repeat 0/1
    """
    mode = (mode or "").strip().lower()

    if mode == "cohort":
        if "cohort" in df_students.columns:
            return (df_students["cohort"].astype(str) == "COVID").astype(int)
        if "hs_graduation_year" in df_students.columns:
            return (df_students["hs_graduation_year"].astype(float) >= 2021).astype(int)
        raise ValueError("cohort 모드에 필요한 컬럼(cohort 또는 hs_graduation_year)이 없습니다.")

    if mode == "any_covid":
        if "any_covid" not in df_students.columns:
            raise ValueError("any_covid 컬럼이 없습니다.")
        return df_students["any_covid"].astype(int)

    if mode == "grade3_covid":
        if "grade3_covid" not in df_students.columns:
            raise ValueError("grade3_covid 컬럼이 없습니다.")
        return df_students["grade3_covid"].astype(int)

    if mode == "remote_days_threshold":
        if "remote_days" not in df_students.columns and "remote_days_json" not in df_students.columns:
            raise ValueError("remote_days(또는 remote_days_json) 컬럼이 없습니다.")
        base = df_students["remote_days"] if "remote_days" in df_students.columns else df_students["remote_days_json"]
        total = base.apply(_parse_remote_days_sum)
        return (total >= int(remote_days_threshold)).astype(int)

    if mode == "repeat":
        if "is_repeat" not in df_students.columns:
            raise ValueError("is_repeat 컬럼이 없습니다.")
        return df_students["is_repeat"].astype(int)

    raise ValueError(f"알 수 없는 group_mode: {mode}")


# ---------------------------------------------------------------------
# 5) Analysis helpers
# ---------------------------------------------------------------------
def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g for two independent samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / pooled
    J = 1 - (3 / (4 * (nx + ny) - 9))  # small sample correction
    return float(J * d)


def two_group_summary(df: pd.DataFrame, y: str, group_col: str = "group") -> Dict:
    """Welch t-test + Brown-Forsythe + Hedges g."""
    g0 = df[df[group_col] == 0][y].dropna().astype(float).to_numpy()
    g1 = df[df[group_col] == 1][y].dropna().astype(float).to_numpy()

    out = {"metric": y, "n0": int(len(g0)), "n1": int(len(g1))}
    if len(g0) < 2 or len(g1) < 2:
        out["error"] = "표본 부족(각 그룹 최소 2개 필요)"
        return out

    bf_stat, bf_p = stats.levene(g0, g1, center="median")  # Brown-Forsythe
    t_stat, p = stats.ttest_ind(g1, g0, equal_var=False)   # group1 - group0

    out.update({
        "mean0": float(np.mean(g0)),
        "mean1": float(np.mean(g1)),
        "diff": float(np.mean(g1) - np.mean(g0)),
        "bf_p": float(bf_p),
        "t": float(t_stat),
        "p": float(p),
        "hedges_g": float(hedges_g(g1, g0)),
    })
    return out


def run_ols(df: pd.DataFrame, y: str, x_cols: List[str]) -> Dict:
    """
    OLS with HC3 robust SE.
    Returns a compact dict with coefficient for 'group' (if present).
    """
    work = df.dropna(subset=[y] + x_cols).copy()
    if len(work) < 10:
        return {"metric": y, "error": "표본 부족(OLS 최소 10행 권장)", "n": int(len(work))}

    X = work[x_cols].astype(float)
    X = sm.add_constant(X, has_constant="add")
    Y = work[y].astype(float)

    model = sm.OLS(Y, X).fit(cov_type="HC3")

    res = {
        "metric": y,
        "n": int(len(work)),
        "r2": float(model.rsquared),
    }

    # store all params (compact)
    for name in model.params.index:
        res[f"b_{name}"] = float(model.params[name])
        res[f"p_{name}"] = float(model.pvalues[name])

    return res


# ---------------------------------------------------------------------
# 6) (Optional) Bootstrap mediation (simple 1 mediator)
#     X=group, M=seteuk_length_mean (or any), Y=overall_volatility/mean
# ---------------------------------------------------------------------
def bootstrap_mediation(df: pd.DataFrame, x: str, m: str, y: str, n_boot: int = 1000, seed: int = 42) -> Dict:
    """
    Simple mediation with bootstrap:
      a: M ~ X + controls
      b,c': Y ~ M + X + controls
      indirect = a*b
    """
    rng = np.random.default_rng(seed)
    work = df.dropna(subset=[x, m, y]).copy()
    if len(work) < 30:
        return {"y": y, "m": m, "error": "표본 부족(bootstrap mediation 최소 30행 권장)", "n": int(len(work))}

    # Standardize for stability (optional)
    work[x] = work[x].astype(float)
    work[m] = work[m].astype(float)
    work[y] = work[y].astype(float)

    # Point estimates
    Xa = sm.add_constant(work[[x]])
    Ma = sm.OLS(work[m], Xa).fit(cov_type="HC3")
    a = float(Ma.params.get(x, np.nan))

    Xb = sm.add_constant(work[[x, m]])
    Yb = sm.OLS(work[y], Xb).fit(cov_type="HC3")
    b = float(Yb.params.get(m, np.nan))
    c_prime = float(Yb.params.get(x, np.nan))
    indirect = a * b

    # Bootstrap CI
    inds = []
    n = len(work)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        samp = work.iloc[idx]

        Xa_s = sm.add_constant(samp[[x]])
        Ma_s = sm.OLS(samp[m], Xa_s).fit()
        a_s = float(Ma_s.params.get(x, np.nan))

        Xb_s = sm.add_constant(samp[[x, m]])
        Yb_s = sm.OLS(samp[y], Xb_s).fit()
        b_s = float(Yb_s.params.get(m, np.nan))

        inds.append(a_s * b_s)

    inds = np.asarray(inds, dtype=float)
    lo, hi = np.nanpercentile(inds, [2.5, 97.5])

    return {
        "y": y,
        "m": m,
        "n": int(n),
        "a": a,
        "b": b,
        "c_prime": c_prime,
        "indirect": float(indirect),
        "indirect_ci_lo": float(lo),
        "indirect_ci_hi": float(hi),
    }


# ---------------------------------------------------------------------
# 7) Orchestrate: Raw -> Processed -> Results
# ---------------------------------------------------------------------
def run_engine(cfg: EngineConfig) -> None:
    np.random.seed(cfg.random_seed)

    out_processed = cfg.out_dir / "processed"
    out_results = cfg.out_dir / "results"
    out_processed.mkdir(parents=True, exist_ok=True)
    out_results.mkdir(parents=True, exist_ok=True)

    # 1) Ingest/Transform
    print("==[1/4] Parse TXT ==")
    df_students, df_seteuk = parse_txt_files(cfg.raw_txt_dir)

    print("==[2/4] Load Grades Excel ==")
    df_grades = load_grades(cfg.grades_excel)

    print("==[3/4] Calculate Volatility ==")
    df_volatility = calculate_volatility(df_students, df_grades)

    # Persist processed
    df_students.to_csv(out_processed / "student_info.csv", index=False, encoding="utf-8-sig")
    df_grades.to_csv(out_processed / "grades.csv", index=False, encoding="utf-8-sig")
    df_seteuk.to_csv(out_processed / "seteuk.csv", index=False, encoding="utf-8-sig")
    df_volatility.to_csv(out_processed / "volatility.csv", index=False, encoding="utf-8-sig")
    print(f"Processed saved -> {out_processed}")

    # 2) Build analysis table (merge)
    print("==[4/4] Analysis ==")
    group = build_group_label(df_students, cfg.group_mode, cfg.remote_days_threshold)
    df_students = df_students.copy()
    df_students["group"] = group

    # Simple seteuk features: per-student mean length and total keywords (if columns exist)
    seteuk_features = None
    if not df_seteuk.empty and "student_id" in df_seteuk.columns:
        tmp = df_seteuk.copy()
        feats = {"student_id": tmp["student_id"].astype(str)}
        if "text_length" in tmp.columns:
            tmp["text_length"] = pd.to_numeric(tmp["text_length"], errors="coerce")
            feats_len = tmp.groupby("student_id", as_index=False)["text_length"].mean().rename(columns={"text_length": "seteuk_length_mean"})
        else:
            feats_len = tmp.groupby("student_id", as_index=False).size().rename(columns={"size": "seteuk_rows"})
            feats_len["seteuk_length_mean"] = np.nan

        # keyword count sum columns (if any)
        kw_cols = [c for c in tmp.columns if c.startswith("keyword_")]
        if kw_cols:
            for c in kw_cols:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0.0)
            feats_kw = tmp.groupby("student_id", as_index=False)[kw_cols].sum()
            feats_kw["seteuk_keyword_total"] = feats_kw[kw_cols].sum(axis=1)
            seteuk_features = feats_len.merge(feats_kw[["student_id", "seteuk_keyword_total"]], on="student_id", how="left")
        else:
            feats_len["seteuk_keyword_total"] = np.nan
            seteuk_features = feats_len

    # Merge base
    df_base = (
        df_students.merge(df_volatility, on="student_id", how="left", suffixes=("", "_vol"))
    )
    if seteuk_features is not None:
        df_base = df_base.merge(seteuk_features, on="student_id", how="left")

    # ---- A) Two-group comparisons
    metrics = [c for c in df_base.columns if c.endswith("_volatility") or c.endswith("_mean")]
    metrics = [m for m in metrics if m in df_base.columns]
    group_rows = [two_group_summary(df_base, y=m, group_col="group") for m in metrics]
    df_group = pd.DataFrame(group_rows).sort_values(["p", "metric"], ascending=[True, True])
    df_group.to_csv(out_results / "group_compare_summary.csv", index=False, encoding="utf-8-sig")

    # ---- B) OLS (y ~ group + controls)
    # Controls: is_repeat, university_admission_year, hs_graduation_year (when available)
    x_cols = ["group"]
    for c in ["is_repeat", "university_admission_year", "hs_graduation_year"]:
        if c in df_base.columns:
            x_cols.append(c)
    ols_rows = [run_ols(df_base, y=m, x_cols=x_cols) for m in metrics]
    df_ols = pd.DataFrame(ols_rows)
    df_ols.to_csv(out_results / "ols_results.csv", index=False, encoding="utf-8-sig")

    # ---- C) Mediation (optional)
    if cfg.run_mediation and "seteuk_length_mean" in df_base.columns:
        med_targets = [m for m in metrics if m in ["overall_volatility", "overall_mean"]]
        med_rows = [
            bootstrap_mediation(
                df_base,
                x="group",
                m="seteuk_length_mean",
                y=y,
                n_boot=cfg.mediation_bootstrap,
                seed=cfg.random_seed,
            )
            for y in (med_targets or metrics[:2])
        ]
        df_med = pd.DataFrame(med_rows)
        df_med.to_csv(out_results / "mediation_results.csv", index=False, encoding="utf-8-sig")

    print(f"Results saved -> {out_results}")
    print("DONE.")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Raw -> Results COVID cohort engine")
    ap.add_argument("--raw_txt_dir", type=str, required=True, help="생활기록부 TXT 폴더 경로")
    ap.add_argument("--grades_excel", type=str, required=True, help="성적 엑셀 파일 경로")
    ap.add_argument("--out_dir", type=str, default="out", help="출력 폴더(out/processed, out/results)")
    ap.add_argument("--group_mode", type=str, default="cohort",
                    choices=["cohort", "any_covid", "grade3_covid", "remote_days_threshold", "repeat"],
                    help="2그룹 분류 기준")
    ap.add_argument("--remote_days_threshold", type=int, default=1, help="remote_days_threshold 모드 임계값")
    ap.add_argument("--run_mediation", action="store_true", help="bootstrap mediation 실행(느림)")
    ap.add_argument("--mediation_bootstrap", type=int, default=1000, help="bootstrap 횟수(기본 1000)")
    ap.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    cfg = EngineConfig(
        raw_txt_dir=Path(args.raw_txt_dir),
        grades_excel=Path(args.grades_excel),
        out_dir=Path(args.out_dir),
        group_mode=args.group_mode,
        remote_days_threshold=args.remote_days_threshold,
        run_mediation=bool(args.run_mediation),
        mediation_bootstrap=int(args.mediation_bootstrap),
        random_seed=int(args.seed),
    )
    run_engine(cfg)


if __name__ == "__main__":
    main()
