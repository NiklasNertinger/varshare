@echo off
echo ===================================================
echo   STARTING OVERNIGHT MT10 PART B (Partial, Scaled, Base)
echo ===================================================

echo [1/3] Running Partial VarShare...
python scripts/run_overnight_4_partial.py

echo [2/3] Running Scaled-Down VarShare...
python scripts/run_overnight_6_scaled.py

echo [3/3] Running Base VarShare (Golden)...
python scripts/run_overnight_1_base.py

echo ===================================================
echo   PART B COMPLETED
echo ===================================================
pause
