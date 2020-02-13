@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/ocr_recognition/." . /s /e /y /d
.\.run_ce.bat
