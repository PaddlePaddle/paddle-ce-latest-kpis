@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/face_detection/." . /s /e /y

.\.run_ce.bat
