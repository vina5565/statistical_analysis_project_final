"""
STEP 3 (FIXED): 올바른 가설 검증 (치명/중요 개선 반영)

H1-1: 코호트 비교 (2021~2024 vs 2018~2020)
      → 기술통계, Welch t-test(기본), Brown-Forsythe(분산), 효과크기,
        OLS(robust HC3) + (선택) 연도 고정효과

H1-2: 매개 분석 (세특 변화/키워드 → 변동성)
      → 다중 매개: Cohort(X) -> M(키워드) -> Y(변동성)
      → b 경로에 X 포함(치명 수정)
      → Sobel 대신 Bootstrap CI(권장, 치명 수정)
      → 결측률(코호트별) 리포트(중요 수정)

주의:
- df_volatility가 학생당 여러 행일 수 있으므로 student_id 기준으로 1행으로 집계(중요 수정)
- F-test는 정규성 가정에 매우 민감하므로 기본적으로 Brown-Forsythe(Levene center=median) 사용
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ----------------------------
# Utils
# ----------------------------
def load_data():
    """데이터 로드"""
    data_dir = Path('data/processed')
    df_students = pd.read_csv(data_dir / 'student_info.csv')
    df_grades = pd.read_csv(data_dir / 'grades.csv')
    df_seteuk = pd.read_csv(data_dir / 'seteuk.csv')
    df_volatility = pd.read_csv(data_dir / 'volatility.csv')
    return df_students, df_grades, df_seteuk, df_volatility


def safe_pct_change(diff, base, eps=1e-9):
    if base is None or np.isnan(base) or abs(base) < eps:
        return np.nan
    return (diff / base) * 100


def ensure_one_row_per_student_volatility(df_volatility, vol_col='overall_volatility'):
    """
    df_volatility가 학생당 여러 행(학기/학년 등)일 가능성이 있으므로 1행으로 집계.
    - 기본: 학생별 mean
    """
    if 'student_id' not in df_volatility.columns:
        raise ValueError("df_volatility에 student_id 컬럼이 없습니다.")

    if vol_col not in df_volatility.columns:
        raise ValueError(f"df_volatility에 {vol_col} 컬럼이 없습니다.")

    # 이미 student_id당 1행이면 그대로
    counts = df_volatility['student_id'].value_counts(dropna=False)
    if (counts <= 1).all():
        return df_volatility.copy()

    df_agg = (df_volatility
              .groupby('student_id', as_index=False)[vol_col]
              .mean())
    return df_agg


def welch_df(x, y):
    """
    Welch-Satterthwaite degrees of freedom
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    num = (vx/nx + vy/ny) ** 2
    den = (vx**2) / (nx**2 * (nx - 1)) + (vy**2) / (ny**2 * (ny - 1))
    if den == 0:
        return np.nan
    return num / den


def hedges_g(x, y):
    """
    Hedges' g (small-sample corrected Cohen's d), pooled SD 기반.
    등분산이 크게 깨질 수 있으니 해석 주의.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if pooled == 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / pooled
    # small sample correction
    J = 1 - (3 / (4*(nx + ny) - 9))
    return d * J


def format_sig(p):
    if p < 0.001:
        return "p < 0.001 ***"
    if p < 0.01:
        return "p < 0.01 **"
    if p < 0.05:
        return "p < 0.05 *"
    return "p >= 0.05 (n.s.)"


def effect_size_label(val):
    if np.isnan(val):
        return "NA"
    a = abs(val)
    if a < 0.2:
        return "작음 (small)"
    if a < 0.5:
        return "중간 (medium)"
    if a < 0.8:
        return "큼 (large)"
    return "매우 큼 (very large)"


def cohort_label(year):
    if 2015 <= year <= 2020:
        return 'Pre-COVID'
    if 2021 <= year <= 2024:
        return 'COVID'
    return 'Other'


# ----------------------------
# H1-1
# ----------------------------
def hypothesis_1_1_cohort(df_students, df_volatility):
    """
    H1-1: 코호트 비교
    - Welch t-test 기본(등분산 가정 회피)
    - Welch df 올바르게 출력(치명 수정)
    - 분산 비교는 Brown-Forsythe(Levene center=median) 기본
    - OLS는 HC3 robust SE 사용(중요 수정)
    - df_volatility 학생당 1행 집계(중요 수정)
    """

    print("="*80)
    print("H1-1: 코호트별 성적 변동성 비교")
    print("  Pre-COVID (2018~2020) vs COVID (2021~2024)")
    print("="*80)

    # 0) 변동성 테이블 학생당 1행 보장
    df_vol = ensure_one_row_per_student_volatility(df_volatility, vol_col='overall_volatility')

    # 1) 병합
    df_merged = df_vol.merge(
        df_students[['student_id', 'hs_graduation_year']],
        on='student_id',
        how='inner'
    )

    # 2) 코호트 정의
    df_merged['cohort'] = df_merged['hs_graduation_year'].apply(cohort_label)

    # 3) 분석 대상
    df_analysis = df_merged[df_merged['cohort'].isin(['Pre-COVID', 'COVID'])].copy()
    df_analysis = df_analysis.dropna(subset=['overall_volatility'])

    pre = df_analysis[df_analysis['cohort'] == 'Pre-COVID']['overall_volatility'].astype(float)
    cov = df_analysis[df_analysis['cohort'] == 'COVID']['overall_volatility'].astype(float)

    print("\n[코호트 구성]")
    print(f"Pre-COVID (2015~2020 졸업): {len(pre)}명")
    for year in range(2016, 2021):
        print(f"  - {year}년: {(df_analysis['hs_graduation_year']==year).sum()}명")

    print(f"\nCOVID (2021~2024 졸업): {len(cov)}명")
    for year in range(2021, 2025):
        c = (df_analysis['hs_graduation_year'] == year).sum()
        if c > 0:
            print(f"  - {year}년: {c}명")

    if len(pre) < 2 or len(cov) < 2:
        print("\nWARNING: 코호트 데이터 부족 (각 코호트 최소 2명 필요)")
        return None

    # 4) 기술통계
    print("\n[1. 기술통계]")
    pre_mean, cov_mean = pre.mean(), cov.mean()
    diff_mean = cov_mean - pre_mean
    diff_pct = safe_pct_change(diff_mean, pre_mean)

    print("\nPre-COVID:")
    print(f"  평균: {pre_mean:.4f}")
    print(f"  표준편차: {pre.std(ddof=1):.4f}")
    print(f"  중앙값: {pre.median():.4f}")
    print(f"  범위: [{pre.min():.4f}, {pre.max():.4f}]")

    print("\nCOVID:")
    print(f"  평균: {cov_mean:.4f}")
    print(f"  표준편차: {cov.std(ddof=1):.4f}")
    print(f"  중앙값: {cov.median():.4f}")
    print(f"  범위: [{cov.min():.4f}, {cov.max():.4f}]")

    if np.isnan(diff_pct):
        print(f"\n평균 차이: {diff_mean:+.4f} (퍼센트 계산 불가: Pre 평균이 0에 근접)")
    else:
        print(f"\n평균 차이: {diff_mean:+.4f} ({diff_pct:+.1f}%)")

    # 5) 분산 비교(기본: Brown-Forsythe)
    print("\n[2. 분산 비교 (Brown-Forsythe / Levene center=median)]")
    bf_stat, bf_p = stats.levene(pre, cov, center='median')
    print(f"  통계량: {bf_stat:.4f}")
    print(f"  p-value: {bf_p:.4f}")
    print(f"  해석: {format_sig(bf_p)} (분산 차이 검정, 정규성에 덜 민감)")

    # 6) t-검정(기본: Welch)
    print("\n[3. 평균 차이 검정 (Welch t-test 기본)]")
    t_stat, p_value = stats.ttest_ind(cov, pre, equal_var=False)  # cov - pre
    df_w = welch_df(cov, pre)

    print(f"  t-통계량: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  자유도(Welch): {df_w:.2f}")
    sig_level = format_sig(p_value)
    print(f"  유의수준: {sig_level}")

    if p_value < 0.05:
        direction = "높음" if cov_mean > pre_mean else "낮음"
        print(f"  CONCLUSION: COVID 코호트의 변동성이 유의미하게 {direction}")
    else:
        print("  CONCLUSION: 두 코호트 간 유의미한 차이 없음")

    # 7) 효과크기(보고용): Hedges g
    g = hedges_g(cov, pre)
    print("\n[4. 효과 크기]")
    print(f"  Hedges' g: {g:.4f}")
    print(f"  해석: {effect_size_label(g)}")
    if bf_p < 0.05:
        print("  NOTE: 분산 차이가 유의하므로(이분산) 효과크기 해석 시 주의하세요.")

    # 8) OLS 회귀(robust HC3)
    print("\n[5. OLS 회귀 분석 (Robust SE: HC3)]")
    df_analysis['is_covid'] = (df_analysis['cohort'] == 'COVID').astype(int)

    # 기본모델
    try:
        model = ols('overall_volatility ~ is_covid', data=df_analysis).fit(cov_type='HC3')
        print("\n모델1: overall_volatility ~ is_covid (HC3)")
        print(f"  R^2: {model.rsquared:.4f}")
        print(f"  is_covid 계수: {model.params['is_covid']:.4f}")
        print(f"  p-value(HC3): {model.pvalues['is_covid']:.4f}")
        ci = model.conf_int().loc['is_covid'].tolist()
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    except Exception as e:
        print(f"  ERROR: 회귀 분석 실패(모델1): {e}")
        model = None

    # 선택: 연도 고정효과(연도별 표본이 충분할 때만)
    # 너무 적으면 오히려 불안정해질 수 있어 자동 조건으로 처리
    try:
        year_counts = df_analysis['hs_graduation_year'].value_counts()
        if (year_counts >= 5).sum() >= 2:  # 최소 2개 연도에서 5명 이상
            fe_model = ols('overall_volatility ~ is_covid + C(hs_graduation_year)', data=df_analysis).fit(cov_type='HC3')
            print("\n모델2(선택): + 졸업연도 고정효과 C(hs_graduation_year) (HC3)")
            print(f"  R^2: {fe_model.rsquared:.4f}")
            print(f"  is_covid 계수: {fe_model.params['is_covid']:.4f}")
            print(f"  p-value(HC3): {fe_model.pvalues['is_covid']:.4f}")
        else:
            fe_model = None
    except Exception as e:
        print(f"  NOTE: 연도 고정효과 모델 스킵/실패: {e}")
        fe_model = None

    # 9) 요약
    print("\n" + "="*80)
    print("H1-1 검증 결과 요약")
    print("="*80)
    print("\n가설: COVID 코호트의 변동성이 Pre-COVID보다 높다")
    print("\n결과:")
    print(f"  - Welch t-test: t={t_stat:.4f}, {sig_level}, df={df_w:.2f}")
    print(f"  - 효과크기(Hedges g): {g:.4f} ({effect_size_label(g)})")
    if np.isnan(diff_pct):
        print(f"  - 평균 차이: {diff_mean:+.4f} (퍼센트: NA)")
    else:
        print(f"  - 평균 차이: {diff_mean:+.4f} ({diff_pct:+.1f}%)")

    hypothesis_supported = (p_value < 0.05) and (diff_mean > 0)
    if hypothesis_supported:
        print("\nRESULT: 가설 채택 (COVID 코호트 변동성 유의미 증가)")
    elif (p_value < 0.05) and (diff_mean < 0):
        print("\nRESULT: 가설 기각 (오히려 COVID 코호트 변동성 감소)")
    else:
        print("\nRESULT: 가설 기각 (유의미한 차이 없음)")

    return {
        'hypothesis': 'H1-1',
        'pre_covid_n': int(len(pre)),
        'covid_n': int(len(cov)),
        'pre_covid_mean': float(pre_mean),
        'covid_mean': float(cov_mean),
        'difference': float(diff_mean),
        'difference_pct': float(diff_pct) if not np.isnan(diff_pct) else np.nan,
        't_statistic_welch': float(t_stat),
        'p_value_welch': float(p_value),
        'welch_df': float(df_w) if not np.isnan(df_w) else np.nan,
        'bf_pvalue': float(bf_p),
        'hedges_g': float(g) if not np.isnan(g) else np.nan,
        'significant': bool(p_value < 0.05),
        'hypothesis_supported': bool(hypothesis_supported)
    }


# ----------------------------
# H1-2 (Bootstrap Mediation)
# ----------------------------
def bootstrap_mediation(df_analysis, m_cols, y_col='overall_volatility', x_col='is_covid',
                        n_boot=5000, seed=42):
    """
    다중 매개 bootstrap:
    - a_k: M_k ~ X
    - b_k: Y ~ X + M_1 + ... + M_k  (치명 수정: X 포함)
    - indirect_k = a_k * b_k
    - total_indirect = sum_k
    - total_effect c: Y ~ X
    - direct_effect c': Y ~ X + M...
    리턴: point estimates + bootstrap CI + (대략적) p-value
    """
    rng = np.random.default_rng(seed)
    df = df_analysis.copy()

    # point estimates
    # a paths
    a = {}
    se_a = {}
    for m in m_cols:
        ma = ols(f'{m} ~ {x_col}', data=df).fit(cov_type='HC3')
        a[m] = ma.params.get(x_col, np.nan)
        se_a[m] = ma.bse.get(x_col, np.nan)

    # b + direct: Y ~ X + Ms
    rhs = ' + '.join([x_col] + m_cols)
    mb = ols(f'{y_col} ~ {rhs}', data=df).fit(cov_type='HC3')
    b = {m: mb.params.get(m, np.nan) for m in m_cols}

    # total effect: Y ~ X
    mc = ols(f'{y_col} ~ {x_col}', data=df).fit(cov_type='HC3')
    total = mc.params.get(x_col, np.nan)
    direct = mb.params.get(x_col, np.nan)

    indirect_point = {m: a[m] * b[m] for m in m_cols}
    total_indirect_point = np.nansum(list(indirect_point.values()))

    # bootstrap
    boot_indirect = {m: [] for m in m_cols}
    boot_total_indirect = []
    boot_direct = []
    boot_total = []

    n = len(df)
    idx = np.arange(n)

    for _ in range(n_boot):
        samp = df.iloc[rng.choice(idx, size=n, replace=True)]

        # a
        a_b = {}
        ok = True
        for m in m_cols:
            try:
                ma_b = ols(f'{m} ~ {x_col}', data=samp).fit()
                a_b[m] = ma_b.params.get(x_col, np.nan)
            except Exception:
                ok = False
                break
        if not ok:
            continue

        # b + direct (X 포함)
        try:
            mb_b = ols(f'{y_col} ~ {rhs}', data=samp).fit()
            b_b = {m: mb_b.params.get(m, np.nan) for m in m_cols}
            direct_b = mb_b.params.get(x_col, np.nan)
        except Exception:
            continue

        # total effect
        try:
            mc_b = ols(f'{y_col} ~ {x_col}', data=samp).fit()
            total_b = mc_b.params.get(x_col, np.nan)
        except Exception:
            total_b = np.nan

        # indirects
        ind_sum = 0.0
        for m in m_cols:
            ind = a_b[m] * b_b[m]
            boot_indirect[m].append(ind)
            ind_sum += ind

        boot_total_indirect.append(ind_sum)
        boot_direct.append(direct_b)
        boot_total.append(total_b)

    def ci(arr, alpha=0.05):
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return (np.nan, np.nan)
        lo = np.quantile(arr, alpha/2)
        hi = np.quantile(arr, 1-alpha/2)
        return (lo, hi)

    def p_two_sided(arr):
        """
        부트스트랩 기반 근사 p-value:
        - 0을 기준으로 부호가 얼마나 뒤집히는지(양측)로 근사
        """
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return np.nan
        prop_pos = np.mean(arr > 0)
        prop_neg = np.mean(arr < 0)
        return 2 * min(prop_pos, prop_neg)

    results = {
        'point': {
            'total_effect_c': total,
            'direct_effect_c_prime': direct,
            'indirect_effects': indirect_point,
            'total_indirect': total_indirect_point,
        },
        'bootstrap': {
            'n_used': int(len(boot_total_indirect)),
            'indirect_ci': {m: ci(boot_indirect[m]) for m in m_cols},
            'indirect_p': {m: p_two_sided(boot_indirect[m]) for m in m_cols},
            'total_indirect_ci': ci(boot_total_indirect),
            'total_indirect_p': p_two_sided(boot_total_indirect),
            'direct_ci': ci(boot_direct),
            'direct_p': p_two_sided(boot_direct),
            'total_ci': ci(boot_total),
            'total_p': p_two_sided(boot_total),
        }
    }
    return results


def hypothesis_1_2_mediation(df_students, df_seteuk, df_volatility,
                            n_boot=5000, seed=42):
    """
    H1-2: 매개 분석 (치명/중요 수정 반영)
    - b 경로에 X 포함(치명)
    - Sobel 대신 bootstrap CI(치명)
    - 결측률 코호트별 출력(중요)
    - df_volatility 학생당 1행 집계(중요)
    """

    print("\n" + "="*80)
    print("H1-2: 매개 분석 (Bootstrap Mediation)")
    print("  경로: Cohort(X) -> 세특 키워드(M) -> 변동성(Y)")
    print("="*80)

    # 0) 변동성 테이블 학생당 1행 보장
    df_vol = ensure_one_row_per_student_volatility(df_volatility, vol_col='overall_volatility')

    # 1) 학생별 세특 키워드 평균(현재 데이터 구조 기준: '변화'가 아니라 '수준'임을 주의)
    #    실제 '변화'를 하려면 기간별/학년별을 나눠 delta를 만들 필요가 있음.
    kw_cols = ['kw_freq_exploration', 'kw_freq_online', 'kw_freq_qualitative']
    kw_cols = [c for c in kw_cols if c in df_seteuk.columns]

    if len(kw_cols) == 0:
        print("ERROR: df_seteuk에 kw_freq_* 컬럼이 없습니다.")
        return None

    seteuk_summary = (df_seteuk
                      .groupby('student_id', as_index=False)[kw_cols]
                      .mean())

    # 2) 병합
    df_merged = (df_vol
                 .merge(df_students[['student_id', 'hs_graduation_year']], on='student_id', how='inner')
                 .merge(seteuk_summary, on='student_id', how='left'))

    # 3) 코호트/더미
    df_merged['cohort'] = df_merged['hs_graduation_year'].apply(cohort_label)
    df_merged['is_covid'] = (df_merged['cohort'] == 'COVID').astype(int)

    df_analysis = df_merged[df_merged['cohort'].isin(['Pre-COVID', 'COVID'])].copy()

    # 4) 결측률 리포트(중요)
    print("\n[결측률(코호트별) 체크]")
    for grp in ['Pre-COVID', 'COVID']:
        sub = df_analysis[df_analysis['cohort'] == grp]
        print(f"\n{grp} (n={len(sub)})")
        miss_y = sub['overall_volatility'].isna().mean() * 100
        print(f"  Y(overall_volatility) 결측률: {miss_y:.1f}%")
        for c in kw_cols:
            miss = sub[c].isna().mean() * 100
            print(f"  M({c}) 결측률: {miss:.1f}%")

    # 5) 분석에 필요한 결측 제거
    needed = ['overall_volatility'] + kw_cols
    df_analysis = df_analysis.dropna(subset=needed).copy()

    print("\n[분석 대상]")
    print(f"전체: {len(df_analysis)}명")
    print(f"  Pre-COVID: {(df_analysis['cohort']=='Pre-COVID').sum()}명")
    print(f"  COVID: {(df_analysis['cohort']=='COVID').sum()}명")

    if len(df_analysis) < 20:
        print("\nWARNING: 표본이 너무 적습니다(권장 최소 20~30명 이상). 결과 해석 주의.")
        if len(df_analysis) < 10:
            print("WARNING: 데이터 부족 (최소 10명 필요)")
            return None

    # 6) 경로별 회귀(출력용, HC3)
    print("\n[1. 경로별 회귀 (HC3)]")

    # a paths: M ~ X
    a_models = {}
    for m in kw_cols:
        ma = ols(f'{m} ~ is_covid', data=df_analysis).fit(cov_type='HC3')
        a_models[m] = ma
        print(f"\n경로 a: is_covid -> {m}")
        print(f"  계수(a): {ma.params.get('is_covid', np.nan):.4f}")
        print(f"  p-value(HC3): {ma.pvalues.get('is_covid', np.nan):.4f}")

    # b + direct: Y ~ X + Ms  (치명 수정: X 포함)
    rhs = ' + '.join(['is_covid'] + kw_cols)
    mb = ols(f'overall_volatility ~ {rhs}', data=df_analysis).fit(cov_type='HC3')

    print("\n경로 b 및 직접효과(c') : Y ~ is_covid + Ms")
    for m in kw_cols:
        print(f"  b({m}): {mb.params.get(m, np.nan):.4f} (p={mb.pvalues.get(m, np.nan):.4f})")
    print(f"  직접효과 c'(is_covid): {mb.params.get('is_covid', np.nan):.4f} (p={mb.pvalues.get('is_covid', np.nan):.4f})")

    # total effect c: Y ~ X
    mc = ols('overall_volatility ~ is_covid', data=df_analysis).fit(cov_type='HC3')
    print("\n경로 c(총효과): Y ~ is_covid")
    print(f"  총효과 c: {mc.params.get('is_covid', np.nan):.4f} (p={mc.pvalues.get('is_covid', np.nan):.4f})")

    # 7) Bootstrap mediation (치명 수정: Sobel 제거)
    print("\n[2. Bootstrap 간접효과 추정]")
    boot = bootstrap_mediation(df_analysis, m_cols=kw_cols,
                              y_col='overall_volatility', x_col='is_covid',
                              n_boot=n_boot, seed=seed)

    n_used = boot['bootstrap']['n_used']
    print(f"  Bootstrap 사용 샘플 수: {n_used} / {n_boot}")
    if n_used < n_boot * 0.8:
        print("  NOTE: 부트스트랩 중 회귀 실패가 많습니다(표본/분산 문제 가능). 결과 해석 주의.")

    point_total = boot['point']['total_effect_c']
    point_direct = boot['point']['direct_effect_c_prime']
    point_total_ind = boot['point']['total_indirect']

    print("\n간접효과(키워드별):")
    for m in kw_cols:
        ind = boot['point']['indirect_effects'][m]
        ci_m = boot['bootstrap']['indirect_ci'][m]
        p_m = boot['bootstrap']['indirect_p'][m]
        print(f"  {m}: {ind:.4f} | 95% CI [{ci_m[0]:.4f}, {ci_m[1]:.4f}] | p≈{p_m:.4f}")

    ci_ti = boot['bootstrap']['total_indirect_ci']
    p_ti = boot['bootstrap']['total_indirect_p']
    print(f"\n총 간접효과: {point_total_ind:.4f} | 95% CI [{ci_ti[0]:.4f}, {ci_ti[1]:.4f}] | p≈{p_ti:.4f}")

    ci_dir = boot['bootstrap']['direct_ci']
    p_dir = boot['bootstrap']['direct_p']
    print(f"직접효과(c'): {point_direct:.4f} | 95% CI [{ci_dir[0]:.4f}, {ci_dir[1]:.4f}] | p≈{p_dir:.4f}")

    ci_tot = boot['bootstrap']['total_ci']
    p_tot = boot['bootstrap']['total_p']
    print(f"총효과(c): {point_total:.4f} | 95% CI [{ci_tot[0]:.4f}, {ci_tot[1]:.4f}] | p≈{p_tot:.4f}")

    # 매개비율(총효과가 0에 가까우면 불안정)
    if point_total is not None and not np.isnan(point_total) and abs(point_total) > 1e-9:
        mediation_ratio = (point_total_ind / point_total) * 100
    else:
        mediation_ratio = np.nan

    # 매개 유의성 판단: 총 간접효과 CI가 0을 포함하는지
    mediation_significant = not (ci_ti[0] <= 0 <= ci_ti[1])

    print("\n" + "="*80)
    print("H1-2 검증 결과 요약")
    print("="*80)
    print("\n가설: 세특 키워드가 코호트 → 변동성 관계를 매개한다")
    print("\n결과(부트스트랩 기준):")
    print(f"  - 총효과(c): {point_total:.4f} (p≈{p_tot:.4f})")
    print(f"  - 직접효과(c'): {point_direct:.4f} (p≈{p_dir:.4f})")
    print(f"  - 총간접효과: {point_total_ind:.4f} (p≈{p_ti:.4f})")
    if not np.isnan(mediation_ratio):
        print(f"  - 매개비율(참고): {mediation_ratio:.1f}% (총효과가 작으면 불안정)")

    if mediation_significant:
        # “완전/부분 매개”는 임계값(0.01) 같은 임의 기준 제거.
        # 직접효과가 유의한지(부트스트랩 CI 0 포함 여부)로 분류(일반적 보고 방식)
        direct_significant = not (ci_dir[0] <= 0 <= ci_dir[1])
        if direct_significant:
            print("\nRESULT: 부분 매개(Partial Mediation) 가능성")
        else:
            print("\nRESULT: 완전 매개(Full Mediation) 가능성")
    else:
        print("\nRESULT: 매개 효과 유의하지 않음(총 간접효과 CI가 0 포함)")

    return {
        'hypothesis': 'H1-2',
        'n': int(len(df_analysis)),
        'total_effect_c': float(point_total) if point_total is not None else np.nan,
        'direct_effect_c_prime': float(point_direct) if point_direct is not None else np.nan,
        'total_indirect': float(point_total_ind) if point_total_ind is not None else np.nan,
        'total_indirect_ci_low': float(ci_ti[0]) if ci_ti[0] is not None else np.nan,
        'total_indirect_ci_high': float(ci_ti[1]) if ci_ti[1] is not None else np.nan,
        'total_indirect_p_approx': float(p_ti) if p_ti is not None else np.nan,
        'direct_ci_low': float(ci_dir[0]) if ci_dir[0] is not None else np.nan,
        'direct_ci_high': float(ci_dir[1]) if ci_dir[1] is not None else np.nan,
        'direct_p_approx': float(p_dir) if p_dir is not None else np.nan,
        'total_p_approx': float(p_tot) if p_tot is not None else np.nan,
        'mediation_ratio_pct': float(mediation_ratio) if not np.isnan(mediation_ratio) else np.nan,
        'mediation_significant': bool(mediation_significant),
        'mediators_used': ",".join(kw_cols)
    }


# ----------------------------
# Main
# ----------------------------
def main():
    print("\n" + "="*80)
    print("STEP 3: 올바른 가설 검증 (FIXED)")
    print("  H1-1: 코호트 비교 (Welch + BF + OLS HC3)")
    print("  H1-2: 매개 분석 (Bootstrap Mediation)")
    print("="*80)

    print("\n데이터 로딩 중...")
    try:
        df_students, df_grades, df_seteuk, df_volatility = load_data()
        print(f"OK: 학생: {len(df_students)}명")
        print(f"OK: 성적: {len(df_grades)}건")
        print(f"OK: 세특: {len(df_seteuk)}건")
        print(f"OK: 변동성: {len(df_volatility)}건")
    except Exception as e:
        print(f"ERROR: 데이터 로드 실패: {e}")
        print("\n먼저 step1_final_complete.py를 실행하세요!")
        return

    results_h1_1 = hypothesis_1_1_cohort(df_students, df_volatility)
    results_h1_2 = hypothesis_1_2_mediation(df_students, df_seteuk, df_volatility,
                                            n_boot=5000, seed=42)

    results_dir = Path('data/results')
    results_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if results_h1_1:
        results.append(results_h1_1)
    if results_h1_2:
        results.append(results_h1_2)

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(results_dir / 'hypothesis_results.csv',
                          index=False, encoding='utf-8-sig')
        print("\nRESULT SAVED: data/results/hypothesis_results.csv")

    print("\n" + "="*80)
    print("가설 검증 완료")
    print("="*80)


if __name__ == "__main__":
    main()
