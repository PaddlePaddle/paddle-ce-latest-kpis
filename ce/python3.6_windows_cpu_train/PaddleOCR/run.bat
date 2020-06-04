@echo off
set models_dir=./../../ocr_repo
rem copy models files
xcopy "%models_dir%/." . /s /e /y /d

if not exist train_data (mklink /j train_data %data_path%\PaddleOCR\train_data)
if not exist pretrain_models (mklink /j pretrain_models %data_path%\PaddleOCR\pretrain_models)
if not exist models (mklink /j models %data_path%\PaddleOCR\models)
set PYTHONPATH=.;%PYTHONPATH%
python -m pip install -r requirments.txt


call .run_ce.bat