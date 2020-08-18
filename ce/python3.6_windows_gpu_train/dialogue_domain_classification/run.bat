@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/dialogue_domain_classification/." . /s /e /y /d
if exist data (
rd /s /q data
)
md data
mklink /j data\input %data_path%\dialogue_domain_classification\data

.\.run_ce.bat
