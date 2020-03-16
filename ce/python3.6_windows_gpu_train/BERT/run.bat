@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/pretrain_language_models/BERT/." . /s /e /y /d

if exist data (
rd /s /q data
mklink /j data %data_path%\dygraph_bert\data
)

.\.run_ce.bat

