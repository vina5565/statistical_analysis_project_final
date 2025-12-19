@echo off
chcp 65001 > nul
echo ===============================================
echo [COVID ENGINE] Raw -> Analysis Results 실행
echo ===============================================

python covid_engine.py ^
  --raw_txt_dir data\raw ^
  --grades_excel "data\raw\학생_성적_정보_양식_통합.xlsx" ^
  --out_dir out ^
  --group_mode cohort

echo.
echo ===============================================
echo 분석 완료! 결과는 out\ 폴더를 확인하세요.
echo ===============================================
pause