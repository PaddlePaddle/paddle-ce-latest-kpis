@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/dialogue_system/\dialogue_general_understanding/." . /s /e /y /d
rd /s /q data
mklink /j data %data_path%\dialogue_understanding\data
.\.run_ce.bat
