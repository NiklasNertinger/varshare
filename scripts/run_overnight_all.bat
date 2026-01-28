@echo off
echo ===================================================
echo   STARTING OVERNIGHT MT10 BENCHMARK (6 ALGOS)
echo ===================================================

echo [1/6] Running Base VarShare...
python scripts/run_overnight_1_base.py

echo [2/6] Running LoMA VarShare...
python scripts/run_overnight_2_lora.py

echo [3/6] Running Reptile VarShare...
python scripts/run_overnight_3_reptile.py

echo [4/6] Running Partial VarShare...
python scripts/run_overnight_4_partial.py

echo [5/6] Running Task Embedding Baseline...
python scripts/run_overnight_5_embedding.py

echo [6/6] Running Scaled-Down VarShare...
python scripts/run_overnight_6_scaled.py

echo ===================================================
echo   ALL EXPERIMENTS COMPLETED. SLEEP WELL! 
echo ===================================================
pause
