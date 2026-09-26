@echo off
title Subaru Emotion Dataset Extractor and Dedicated FAISS Index Builder
color 0A
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1

echo ========================================================
echo   SUBARU EMOTION DATASET EXTRACTOR and FAISS BUILDER
echo   Target: 3,733 Sliced Sean Chiplock Vocal Segments
echo   Extracting Rage, Grief, Comedic, Cold Threat Indices
echo   Running in External Desktop Terminal
echo ========================================================
echo.

"a:\Projects\novel reader\voice-server\.venv\Scripts\python.exe" -u "a:\Projects\novel-reader-voice-engine\experiments\subaru\extract_and_build_emotion_indices.py"

echo.
echo ========================================================
echo   Emotion Index Extraction Complete!
echo ========================================================
echo.
pause
