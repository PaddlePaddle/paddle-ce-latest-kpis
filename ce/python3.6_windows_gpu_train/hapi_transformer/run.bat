@echo off
@echo off
set models_dir=./../../hapi_repo
rem copy models files
xcopy "%models_dir%/examples/transformer/." . /s /e /y /d
if not exist data (mklink /j data  %data_path%\transformer)

.\.run_ce.bat

