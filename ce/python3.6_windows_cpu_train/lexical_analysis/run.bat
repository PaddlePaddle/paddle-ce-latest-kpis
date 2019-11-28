@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/lexical_analysis/." . /s /e /y /d
if not exist data ( mklink /j data %data_path%\lexical_analysis\data )
if not exist pretrained ( mklink /j pretrained  %data_path%\lexical_analysis\pretrained )
if not exist model_baseline (mklink /j model_baseline %data_path%\lexical_analysis\model_baseline)
if not exist model_finetuned (mklink /j model_finetuned %data_path%\lexical_analysis\model_finetuned)

.\.run_ce.bat
