@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleRec/ctr/deepfm_dygraph/." . /s /e /y /d

rd /s /q data
mklink /j data %data_path%\dygraph_deepfm\data

.\.run_ce.bat
