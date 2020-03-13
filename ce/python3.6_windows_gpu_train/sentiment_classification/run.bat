@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/sentiment_classification/." . /s /e /y /d

if not exist senta_data (mklink /j senta_data  %data_path%\sentiment_classification\senta_data)
if not exist senta_model (mklink /j  senta_model  %data_path%\sentiment_classification\senta_model) 

mklink /j "../shared_modules" "%models_dir%\PaddleNLP\shared_modules"
call .run_ce.bat
rd /s /q "../shared_modules"
