@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/dygraph/bert/." . /s /e /y /d
if not exist data (
mklink /j data %data_path%\dygraph_bert\data
)


.\.run_ce.bat
