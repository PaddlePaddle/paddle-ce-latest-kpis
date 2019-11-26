@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleRec/din/." . /s /e /y
if exist data  (rd /s /q data)
mklink /j data %data_path%\din
.\.run_ce.bat
