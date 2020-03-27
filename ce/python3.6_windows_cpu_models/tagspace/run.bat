@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleRec/tagspace/." . /s /e /y /d
if not exist train_data (mklink /j train_data %data_path%\\full_data\tagspace\train_data)
if not exist test_data (mklink /j test_data %data_path%\full_data\tagspace\test_data)
if not exist vocab_text.txt (mklink /h vocab_text.txt %data_path%\full_data\tagspace\vocab_text.txt)
if not exist vocab_tag.txt (mklink /h vocab_tag.txt %data_path%\full_data\tagspace\vocab_tag.txt)
.\.run_ce.bat
