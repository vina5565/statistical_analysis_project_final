# COVID-19 성적 분석 파이프라인

본 프로젝트는 생활기록부(TXT)와 성적 엑셀 파일을 입력으로 받아  
코로나19 전후 학생 집단 간 **성적 수준 및 성적 변동성의 구조적 차이**를 분석하는 파이프라인이다.

단순 평균 비교를 넘어,  
- 집단 간 평균 차이
- 성적 변동성(불안정성)
- 통제변수를 고려한 회귀 분석

을 통해 코로나19 시기의 교육 환경 변화가 학생 성취에 미친 영향을 종합적으로 검토한다.

---

## 1. 프로젝트 구조
Statistical_Analysis_Project/
├─ covid_engine.py # Raw → 분석 결과 생성 엔진
├─ print_interpretation.py # 분석 결과 터미널 출력용 스크립트
│
├─ data/
│ └─ raw/
│ ├─ *.txt # 생활기록부 TXT 파일
│ └─ 학생_성적_정보_양식_통합.xlsx
│
├─ out/
│ ├─ processed/ # 중간 가공 데이터
│ └─ results/ # 최종 분석 결과
│
└─ scripts/
├─ run_engine.bat # 분석 실행 (Windows)
└─ run_interpret.bat # 결과 해석 출력 (Windows)
---

## 2. 실행 방법 (Windows 기준)

### ① 분석 실행 (Raw → 결과 생성)
다음 파일을 **더블클릭**하여 실행한다.
- 입력:
  - `data/raw/` 내 생활기록부 TXT
  - 성적 엑셀 파일
- 출력:
  - `out/processed/` : 가공된 데이터
  - `out/results/`   : 분석 결과 CSV

---

### ② 결과 해석 출력
다음 파일을 **더블클릭**하여 실행한다.
- 터미널에 다음 정보가 자동 출력된다:
  - 어떤 지표(metric)에서
  - 어떤 통계 방법을 사용했고
  - 코로나 그룹과 비코로나 그룹 간 어떤 차이가 있었는지

---

## 3. 터미널에서 직접 실행 (선택)

```bash
python covid_engine.py \
  --raw_txt_dir data/raw \
  --grades_excel data/raw/학생_성적_정보_양식_통합.xlsx \
  --out_dir out \
  --group_mode cohort

python print_interpretation.py --results_dir out/results
## 4. 그룹 정의 옵션 (실험 조건)

본 프로젝트는 코로나19의 영향을 다양한 기준으로 정의하기 위해  
여러 가지 그룹 분류 방식을 제공한다.  
분석 실행 시 `--group_mode` 옵션을 통해 아래 중 하나를 선택할 수 있다.

| 옵션 | 설명 |
|---|---|
| `cohort` | 졸업 연도를 기준으로 코로나 코호트와 비코로나 코호트 구분 |
| `any_covid` | 재학 중 원격수업 경험 여부 기준 |
| `grade3_covid` | 3학년 시기에 코로나 영향을 받았는지 여부 기준 |
| `remote_days_threshold` | 원격수업 일수가 특정 임계값 이상인지 여부 기준 |

예시:
```bash
python covid_engine.py ... --group_mode any_covid

5. 결과 해석 기준 (분석 방법)

본 프로젝트는 단일 통계 기법에 의존하지 않고,
여러 분석 방법을 병행하여 결과의 신뢰성과 해석 가능성을 높인다.

사용된 분석 방법은 다음과 같다.

집단 간 평균 차이 검정 :
- Welch t-test
(집단 간 분산이 다를 수 있음을 고려한 평균 비교)

집단 간 분산 차이 검정 :
- Brown–Forsythe test
(성적 변동성 차이 검증)

효과크기 산출 :
- Hedges’ g
(표본 수 차이를 보정한 효과크기 지표)

통제 분석 :
- OLS 회귀 분석
- Robust standard error (HC3)

이를 통해 단순한 평균 성적 차이뿐 아니라
성적의 안정성(변동성) 및
통제변수를 고려한 구조적 차이를 함께 분석한다.

6. 재현성

본 프로젝트의 모든 분석 과정은
명시적인 CLI 인자 기반 실행 방식으로 설계되었다.

동일한 입력 데이터 / 동일한 실행 옵션을 사용할 경우, 언제든지 동일한 결과를 재현할 수 있다.

또한 분석 실행은 특정 IDE에 의존하지 않으며,
터미널 또는 스크립트 실행을 통해 운영체제 환경 차이를 최소화한다.