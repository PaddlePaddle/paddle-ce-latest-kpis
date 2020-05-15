@echo off
set models_dir=./../../hapi_repo
rem copy models files
xcopy "%models_dir%/examples/cyclegan/." . /s /e /y /d
if not exist data  (mklink /j  data  %data_path%\gan)
python -m pip install -U scipy==1.2.0
.\.run_ce.bat
