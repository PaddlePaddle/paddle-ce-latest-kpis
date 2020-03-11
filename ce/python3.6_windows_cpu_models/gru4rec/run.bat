@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleRec/gru4rec/." . /s /e /y /d
rd /s /q train_data
rd /s /q test_data
del vocab.txt
mklink /j train_data %data_path%\full_data\gru4rec\train_data
mklink /j test_data %data_path%\full_data\gru4rec\test_data
mklink /h vocab.txt %data_path%\full_data\gru4rec\vocab.txt
.\.run_ce.bat
