@echo off
title Subaru Emotion-Aware Voice Generator (External Terminal)
color 0A
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo ========================================================
echo   SUBARU EMOTION-AWARE VOICE GENERATOR
echo   Using Dedicated Emotional FAISS Indices:
echo    - subaru_rage_v2.index
echo    - subaru_grief_v2.index
echo    - subaru_comedic_v2.index
echo    - subaru_cold_threat_v2.index
echo    - subaru_neutral_v2.index
echo ========================================================
echo.

"a:\Projects\novel reader\voice-server\.venv\Scripts\python.exe" -u "a:\Projects\novel-reader-voice-engine\experiments\subaru\run_experiment.py"

echo.
echo ========================================================
echo   Generation Complete!
echo ========================================================
echo.
pause
