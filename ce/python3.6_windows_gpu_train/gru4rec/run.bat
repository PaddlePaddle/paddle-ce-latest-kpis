@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleRec/gru4rec/." . /s /e /y

.\.run_ce.bat
