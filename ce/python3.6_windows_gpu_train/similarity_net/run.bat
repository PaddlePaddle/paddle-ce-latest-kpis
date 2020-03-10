@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/similarity_net/." . /s /e /y /d
if not exist data (mklink /j data  %data_path%\similarity_net\data)
if not exist model_files (mklink /j model_files  %data_path%\similarity_net\model_files)
mklink /j "../shared_modules" "%models_dir%\PaddleNLP\shared_modules"
call .run_ce.bat
rd /s /q "../shared_modules"
