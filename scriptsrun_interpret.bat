@echo off
chcp 65001 > nul
echo ===============================================
echo [INTERPRETATION] 결과 해석 출력
echo ===============================================

python print_interpretation.py ^
  --results_dir out\results ^
  --top_k 10 ^
  --alpha 0.05

echo.
echo ===============================================
echo 해석 출력 완료!
echo ===============================================
pause
