@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/gan/." . /s /e /y /d
if exist data (rd /s /q data)
mklink /j data %data_path%\gan
.\.run_ce.bat
