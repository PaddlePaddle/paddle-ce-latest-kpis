@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/PaddleDialogue/auto_dialogue_evaluation/." . /s /e /y

.\.run_ce.bat
