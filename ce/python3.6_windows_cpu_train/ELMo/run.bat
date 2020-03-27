@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/pretrain_language_models/ELMo/." . /s /e /y /d

.\.run_ce.bat
