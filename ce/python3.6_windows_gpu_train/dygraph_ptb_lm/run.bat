@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/dygraph/ptb_lm/." . /s /e /y /d
cd data
if not exist simple-examples (mklink /j simple-examples %data_path%\simple-examples)
cd ..
.\.run_ce.bat
