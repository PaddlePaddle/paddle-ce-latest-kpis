@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleRec/word2rec/." . /s /e /y /d
if not exist data (
mklink /j data %data_path%\word2vec
)

.\.run_ce.bat
