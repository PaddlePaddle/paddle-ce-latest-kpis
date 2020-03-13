@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/dygraph/lac/." . /s /e /y /d
if not exist data ( mklink /j data %data_path%\lexical_analysis\data )
.\.run_ce.bat
