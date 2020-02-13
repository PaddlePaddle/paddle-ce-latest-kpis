@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/PaddleDialogue/auto_dialogue_evaluation/." . /s /e /y /d
rd /s /q data
mklink /j data %data_path%\auto_dialogue_evaluation\data
.\.run_ce.bat
