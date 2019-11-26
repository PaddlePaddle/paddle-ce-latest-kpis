@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/language_model/." . /s /e /y
cd data
if not exist simple-examples (mklink /j simple-examples %data_path%\simple-examples)
cd ..
.\.run_ce.bat
