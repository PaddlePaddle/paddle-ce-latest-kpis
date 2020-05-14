@echo off
set models_dir=./../../hapi_repo
rem copy models files
xcopy "%models_dir%/examples/seq2seq/." . /s /e /y /d
if not exist data (mklink /j data %data_path%\PaddleTextGEN\seq2seq)
call .run_ce.bat

