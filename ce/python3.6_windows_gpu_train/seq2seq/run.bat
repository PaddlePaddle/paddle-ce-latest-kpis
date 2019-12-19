@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/PaddleTextGEN/seq2seq/." . /s /e /y /d
if not exist data (mklink /j data %data_path%\PaddleTextGEN\seq2seq)
call .run_ce.bat

