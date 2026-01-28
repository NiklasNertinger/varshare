@echo off
echo ===================================================
echo   STARTING OVERNIGHT MT10 PART A (LoMA, Reptile, Embedding)
echo ===================================================

echo [1/3] Running LoMA VarShare...
python scripts/run_overnight_2_lora.py

echo [2/3] Running Reptile VarShare...
python scripts/run_overnight_3_reptile.py

echo [3/3] Running Embedding Baseline...
python scripts/run_overnight_5_embedding.py

echo ===================================================
echo   PART A COMPLETED
echo ===================================================
pause
