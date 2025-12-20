"""
print_interpretation.py
covid_engine 결과(또는 interpret_results.py가 만든 요약 엑셀)를 읽어서
터미널에 "지표 / 방법 / 결과"를 자동 출력하는 스크립트.

지원 입력 (둘 중 하나만 있어도 됨):
1) results 폴더의 CSV
   - group_compare_summary.csv
   - ols_results.csv

2) interpret_results.py가 만든 엑셀
   - interpretation_summary.xlsx (시트: GroupCompare, OLS)

사용 예시:
  python print_interpretation.py --results_dir "c:/Python/statistical_analysis/out/results"
  python print_interpretation.py --excel "c:/Python/statistical_analysis/out/results/interpretation_summary.xlsx"

옵션:
  --alpha 0.05  (유의수준)
  --top_k 10    (상위 몇 개 지표 출력)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import re
import numpy as np
from scipy import stats


def _direction_from_diff(diff: float) -> str:
    # diff = mean1(코로나) - mean0(비코로나)
    if pd.isna(diff):
        return "해석불가"
    if diff > 0:
        return "코로나 그룹이 더 큼(↑)"
    if diff < 0:
        return "코로나 그룹이 더 작음(↓)"
    return "차이 없음"


def _effect_label(g: float) -> str:
    if pd.isna(g):
        return "N/A"
    ag = abs(g)
    if ag < 0.2:
        return "매우 작음"
    if ag < 0.5:
        return "작음"
    if ag < 0.8:
        return "중간"
    return "큼"

# -----------------------------
# 세특 키워드 사전(필요시 수정)
# -----------------------------
KEYWORDS = {
    "exp_inquiry": [
        "실험", "탐구", "탐색", "가설", "검증", "관찰", "측정", "분석", "결과", "변인",
        "연구", "프로젝트", "설계", "자료수집", "데이터수집", "보고서", "발표", "토의"
    ],
    "online_data": [
        "온라인", "비대면", "원격", "줌", "zoom", "구글", "클래스룸", "lms",
        "데이터", "코딩", "파이썬", "python", "엑셀", "스프레드시트", "통계", "시각화",
        "모델", "머신러닝", "ai", "크롤링", "설문", "폼", "form"
    ],
}

def _find_text_column(df: pd.DataFrame) -> str:
    """seteuk.csv에서 텍스트 컬럼 자동 탐지"""
    candidates = ["text", "content", "seteuk_text", "body", "memo", "desc", "sentence"]
    for c in candidates:
        if c in df.columns:
            return c
    # object 컬럼 중 하나 선택
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError("seteuk.csv에서 텍스트 컬럼을 찾을 수 없습니다.")
    return obj_cols[0]

def _normalize_text(s) -> str:
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_sid(x) -> str:
    """
    student_id 정규화:
    - 201611091.0 -> 201611091
    - 공백 제거
    - 혹시 섞인 문자 있으면 숫자만 추출
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)           # float로 읽힌 학번 처리
    m = re.search(r"(\d{6,})", s)        # 최소 6자리 이상 숫자 추출
    return m.group(1) if m else s


def _keyword_count(text: str, keywords: list[str]) -> int:
    if not text:
        return 0
    cnt = 0
    for kw in keywords:
        cnt += text.count(kw.lower())
    return cnt

def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
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
    J = 1 - (3 / (4 * (nx + ny) - 9))
    return float(J * d)

def _two_group_stats(df: pd.DataFrame, metric: str, group_col: str = "group") -> dict:
    """Welch t-test + Brown-Forsythe + Hedges g (diff=mean1-mean0)"""
    g0 = df[df[group_col] == 0][metric].dropna().astype(float).to_numpy()
    g1 = df[df[group_col] == 1][metric].dropna().astype(float).to_numpy()

    # ✅ 어떤 경우에도 컬럼이 존재하도록 기본값 채워두기
    out = {
        "metric": metric,
        "n0": int(len(g0)),
        "n1": int(len(g1)),
        "mean0": np.nan,
        "mean1": np.nan,
        "diff": np.nan,
        "bf_p": np.nan,
        "p": np.nan,
        "hedges_g": np.nan,
        "error": "",
    }

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
        "p": float(p),
        "hedges_g": float(_hedges_g(g1, g0)),
        "error": "",
    })
    return out




def load_from_excel(xlsx_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_group = pd.read_excel(xlsx_path, sheet_name="GroupCompare")
    df_ols = pd.read_excel(xlsx_path, sheet_name="OLS")
    return df_group, df_ols


def load_from_csv(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_path = results_dir / "group_compare_summary.csv"
    ols_path = results_dir / "ols_results.csv"
    if not group_path.exists():
        raise FileNotFoundError(f"파일 없음: {group_path}")
    if not ols_path.exists():
        raise FileNotFoundError(f"파일 없음: {ols_path}")
    df_group = pd.read_csv(group_path)
    df_ols = pd.read_csv(ols_path)
    return df_group, df_ols


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_group_compare(df_group_raw: pd.DataFrame, alpha: float, top_k: int) -> None:
    df = df_group_raw.copy()

    # 안전 처리
    for c in ["p", "bf_p", "diff", "hedges_g", "mean0", "mean1"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["direction"] = df.get("diff").apply(_direction_from_diff)
    df["effect"] = df.get("hedges_g").apply(_effect_label)
    df["sig"] = df.get("p").lt(alpha)

    df = df.sort_values(["p", "metric"], ascending=[True, True])

    print_header("1) 2그룹 비교 결과 (Welch t-test + Brown-Forsythe + Hedges' g)")
    print(f"- 유의수준(alpha) = {alpha}")
    print("- diff = mean1(코로나) - mean0(비코로나)")
    print("- p = Welch t-test p-value / bf_p = 분산 차이(Brown-Forsythe) p-value")
    print()

    sig_df = df[df["sig"] == True].head(top_k)

    if sig_df.empty:
        print(f"※ p < {alpha} 기준으로 유의한 지표가 없습니다.")
        return

    print(f"※ 유의 지표 TOP {min(top_k, len(sig_df))}")
    for _, r in sig_df.iterrows():
        metric = r.get("metric", "N/A")
        n0, n1 = r.get("n0", "N/A"), r.get("n1", "N/A")
        mean0, mean1 = r.get("mean0", float("nan")), r.get("mean1", float("nan"))
        diff = r.get("diff", float("nan"))
        p = r.get("p", float("nan"))
        bf_p = r.get("bf_p", float("nan"))
        g = r.get("hedges_g", float("nan"))

        print(f"- 지표: {metric}")
        print(f"  · 방법: Welch t-test(평균 차이), Brown-Forsythe(분산 차이), Hedges' g(효과크기)")
        print(f"  · 표본: n0={n0}, n1={n1}")
        print(f"  · 평균: mean0={mean0:.4g}, mean1={mean1:.4g}, diff={diff:.4g} ({r.get('direction')})")
        print(f"  · 유의성: p={p:.4g}, bf_p={bf_p:.4g}")
        print(f"  · 효과크기: g={g:.3g} ({r.get('effect')})")
        print()


def print_ols(df_ols_raw: pd.DataFrame, alpha: float, top_k: int) -> None:
    df = df_ols_raw.copy()

    # 안전 처리
    for c in ["p_group", "b_group", "r2", "n"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "b_group" not in df.columns or "p_group" not in df.columns:
        print_header("2) 회귀 결과 (OLS + HC3 robust)")
        print("※ 이 파일에는 b_group/p_group 컬럼이 없어 OLS 해석을 출력할 수 없습니다.")
        return

    df["direction"] = df["b_group"].apply(
        lambda v: "코로나 그룹 증가(↑)" if pd.notna(v) and v > 0 else ("코로나 그룹 감소(↓)" if pd.notna(v) and v < 0 else "0 또는 N/A")
    )
    df["sig"] = df["p_group"].lt(alpha)
    df = df.sort_values(["p_group", "metric"], ascending=[True, True], na_position="last")

    print_header("2) 회귀 결과 (OLS + HC3 robust)")
    print(f"- 유의수준(alpha) = {alpha}")
    print("- b_group: (코로나=1)일 때 Y가 얼마나 변하는지(통제변수 포함)")
    print("- p_group: 그룹 효과 유의성")
    print()

    sig_df = df[df["sig"] == True].head(top_k)
    if sig_df.empty:
        print(f"※ p_group < {alpha} 기준으로 유의한 지표가 없습니다.")
        return

    print(f"※ 그룹 효과 유의 지표 TOP {min(top_k, len(sig_df))}")
    for _, r in sig_df.iterrows():
        metric = r.get("metric", "N/A")
        n = int(r.get("n", 0)) if pd.notna(r.get("n", 0)) else 0
        r2 = r.get("r2", float("nan"))
        b = r.get("b_group", float("nan"))
        p = r.get("p_group", float("nan"))
        direction = r.get("direction", "N/A")

        print(f"- 지표: {metric}")
        print(f"  · 방법: OLS 회귀 + robust SE(HC3)")
        print(f"  · 결과: b_group={b:.4g} ({direction}), p_group={p:.4g}, R²={r2:.3g}, n={n}")
        print()

def print_seteuk_keyword_analysis(results_dir: Path, alpha: float, top_k: int) -> None:
    """
    out/processed/seteuk.csv + student_info.csv를 읽어서
    세특 키워드(이미 계산된 컬럼)를 학생별로 집계한 뒤 2그룹 비교 결과 출력
    """
    processed_dir = results_dir.parent / "processed"  # out/results -> out/processed
    seteuk_path = processed_dir / "seteuk.csv"
    students_path = processed_dir / "student_info.csv"

    print_header("3) 세특 키워드 빈도 비교 (seteuk.csv 기반)")

    if not seteuk_path.exists() or not students_path.exists():
        print("※ 필요한 파일이 없습니다.")
        print(f"  - seteuk.csv: {seteuk_path}")
        print(f"  - student_info.csv: {students_path}")
        return

    df_seteuk = pd.read_csv(seteuk_path)
    df_students = pd.read_csv(students_path)

    # --- group 만들기(없으면 cohort/졸업연도 기반)
    if "group" not in df_students.columns:
        if "cohort" in df_students.columns:
            df_students["group"] = (df_students["cohort"].astype(str) == "COVID").astype(int)
        elif "hs_graduation_year" in df_students.columns:
            df_students["group"] = (pd.to_numeric(df_students["hs_graduation_year"], errors="coerce") >= 2021).astype(int)
        else:
            print("※ student_info.csv에 group/cohort/hs_graduation_year 컬럼이 없어 그룹을 만들 수 없습니다.")
            return

    if "student_id" not in df_seteuk.columns or "student_id" not in df_students.columns:
        print("※ student_id 컬럼이 없어 병합할 수 없습니다.")
        return

    # ✅ 핵심: student_id 정규화(201611091.0 문제 해결)
    df_students["student_id"] = df_students["student_id"].apply(normalize_sid)
    df_seteuk["student_id"] = df_seteuk["student_id"].apply(normalize_sid)

    # --- seteuk.csv가 이미 키워드 카운트/빈도 컬럼을 갖고 있으면 그걸 그대로 사용
    required_cols = {
        "exploration_keyword_count",
        "online_keyword_count",
        "qualitative_keyword_count",
        "kw_freq_exploration",
        "kw_freq_online",
        "kw_freq_qualitative",
        "content_length",
    }

    if not required_cols.issubset(set(df_seteuk.columns)):
        # fallback: 기존 텍스트 기반(너 KEYWORDS 로직)으로 계산
        # (근데 너 CSV는 이미 컬럼이 있으니 보통 여기 안 들어감)
        text_col = _find_text_column(df_seteuk)
        df_seteuk[text_col] = df_seteuk[text_col].map(_normalize_text)

        agg = df_seteuk.groupby("student_id")[text_col].apply(lambda x: " ".join(x)).reset_index()
        agg["text_len"] = agg[text_col].str.len().astype(float).clip(lower=1.0)

        for key, kws in KEYWORDS.items():
            agg[f"{key}_cnt"] = agg[text_col].apply(lambda t: _keyword_count(t, kws)).astype(float)
            agg[f"{key}_per1k"] = (agg[f"{key}_cnt"] / agg["text_len"]) * 1000.0

        df = df_students[["student_id", "group"]].merge(agg, on="student_id", how="inner")

        metrics = ["exp_inquiry_cnt", "exp_inquiry_per1k", "online_data_cnt", "online_data_per1k", "text_len"]

    else:
        # ✅ 추천 루트: seteuk.csv의 계산 결과를 학생별로 집계
        per_student = df_seteuk.groupby("student_id", as_index=False).agg({
            "exploration_keyword_count": "sum",
            "online_keyword_count": "sum",
            "qualitative_keyword_count": "sum",
            "kw_freq_exploration": "mean",
            "kw_freq_online": "mean",
            "kw_freq_qualitative": "mean",
            "content_length": "sum",
        })

        # 병합
        df = df_students[["student_id", "group"]].merge(per_student, on="student_id", how="inner")

        # 비교할 지표
        metrics = [
            "exploration_keyword_count",
            "online_keyword_count",
            "qualitative_keyword_count",
            "kw_freq_exploration",
            "kw_freq_online",
            "kw_freq_qualitative",
            "content_length",
        ]

    # ---- DEBUG (원하면 지워도 됨)
    print(f"[DEBUG] students={len(df_students)}, seteuk={len(df_seteuk)}, merged={len(df)}")
    if len(df) == 0:
        print("※ 병합 결과가 0행입니다. student_id 정규화가 여전히 맞지 않거나, 두 파일의 학번 집합이 다릅니다.")
        print("  - student_info student_id 예시:", df_students["student_id"].head(5).tolist())
        print("  - seteuk student_id 예시:", df_seteuk["student_id"].head(5).tolist())
        return

    # ---- stats
    rows = []
    for m in metrics:
        if m in df.columns:
            rows.append(_two_group_stats(df, m, group_col="group"))

    out = pd.DataFrame(rows)

    out["sig"] = pd.to_numeric(out["p"], errors="coerce").lt(alpha)
    out = out.sort_values(["p", "metric"], ascending=[True, True])

    sig_df = out[out["sig"] == True].head(top_k)

    print(f"- 유의수준(alpha) = {alpha}")
    print("- diff = mean1(코로나) - mean0(비코로나)")
    print("- p = Welch t-test p-value / bf_p = 분산 차이(Brown-Forsythe) p-value")
    print()

    if sig_df.empty:
        print(f"※ p < {alpha} 기준으로 유의한 세특 키워드 지표가 없습니다.")
        # 그래도 상위 7개 정도 요약 출력
        for _, r in out.head(min(7, len(out))).iterrows():
            print(f"- {r['metric']}: p={r['p']:.4g}, diff={r['diff']:.4g}")
    else:
        print(f"※ 유의 지표 TOP {min(top_k, len(sig_df))}")
        for _, r in sig_df.iterrows():
            metric = r["metric"]
            n0, n1 = r["n0"], r["n1"]
            mean0, mean1 = r["mean0"], r["mean1"]
            diff = r["diff"]
            p = r["p"]
            bf_p = r["bf_p"]
            g = r["hedges_g"]

            direction = _direction_from_diff(diff)
            effect = _effect_label(g)

            print(f"- 지표: {metric}")
            print(f"  · 방법: Welch t-test(평균 차이), Brown-Forsythe(분산 차이), Hedges' g(효과크기)")
            print(f"  · 표본: n0={n0}, n1={n1}")
            print(f"  · 평균: mean0={mean0:.4g}, mean1={mean1:.4g}, diff={diff:.4g} ({direction})")
            print(f"  · 유의성: p={p:.4g}, bf_p={bf_p:.4g}")
            print(f"  · 효과크기: g={g:.3g} ({effect})")
            print()

    # 저장
    save_path = results_dir / "seteuk_keyword_group_compare.csv"
    out.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"(저장) {save_path}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, default="", help="out/results 폴더 경로(기본: out/results)")
    ap.add_argument("--excel", type=str, default="", help="interpretation_summary.xlsx 경로(있으면 이걸 우선 사용)")
    ap.add_argument("--alpha", type=float, default=0.05, help="유의수준")
    ap.add_argument("--top_k", type=int, default=10, help="상위 몇 개 지표 출력")
    args = ap.parse_args()

    alpha = float(args.alpha)
    top_k = int(args.top_k)

    if args.excel:
        xlsx_path = Path(args.excel)
        if not xlsx_path.exists():
            raise FileNotFoundError(f"엑셀 파일이 없습니다: {xlsx_path}")
        df_group, df_ols = load_from_excel(xlsx_path)
    else:
        results_dir = Path(args.results_dir) if args.results_dir else Path("out/results")
        df_group, df_ols = load_from_csv(results_dir)

    print_header("COVID ENGINE 결과 요약 출력")
    print("이 스크립트는 '지표 / 사용한 방법 / 결과'를 터미널로 자동 요약합니다.")
    print()

    print_group_compare(df_group, alpha=alpha, top_k=top_k)
    print_ols(df_ols, alpha=alpha, top_k=top_k)
    base_results_dir = results_dir if not args.excel else Path(args.excel).parent
    print_seteuk_keyword_analysis(base_results_dir, alpha=alpha, top_k=top_k)

    print("=" * 80)
    print("끝.")
    print("=" * 80)


if __name__ == "__main__":
    main()
