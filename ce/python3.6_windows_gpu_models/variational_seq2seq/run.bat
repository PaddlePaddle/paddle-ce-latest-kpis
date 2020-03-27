@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleNLP/seq2seq/variational_seq2seq/." . /s /e /y /d
if not exist data (mklink /j data %data_path%\PaddleTextGEN\variational_seq2seq)
call .run_ce.bat

