@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/PaddleGAN/." . /s /e /y /d
if not exist data (mklink /j data %data_path%\gan)
.\.run_ce.bat
