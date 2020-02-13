@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleRec/ctr/dnn/." . /s /e /y /d
rd /s /q data
mklink /j data %data_path%\ctr\dnn
pip install -r requirements.txt
.\.run_ce.bat
