@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/icnet/." . /s /e /y /d

.\.run_ce.bat
