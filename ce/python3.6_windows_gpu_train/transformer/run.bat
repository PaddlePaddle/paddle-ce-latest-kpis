@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/machine_reading_comprehension/transformer/." . /s /e /y /d
if not exist data (mklink /j data  %data_path%\transformer)

.\.run_ce.bat

