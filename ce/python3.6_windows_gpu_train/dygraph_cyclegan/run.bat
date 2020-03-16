@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/dygraph/cycle_gan/." . /s /e /y /d
if not exist data (
mklink /j data %data_path%\gan
)
pip install -U scipy==1.2.1

.\.run_ce.bat
