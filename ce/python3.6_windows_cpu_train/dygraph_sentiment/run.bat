@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/dygraph/sentiment/." . /s /e /y /d
if not exist senta_data (mklink /j senta_data %data_path%\sentiment_classification\senta_data)
.\.run_ce.bat
