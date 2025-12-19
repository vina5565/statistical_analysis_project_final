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

    print("=" * 80)
    print("끝.")
    print("=" * 80)


if __name__ == "__main__":
    main()
