@echo off
set models_dir=./../../hapi_repo
rem copy models files
xcopy "%models_dir%/examples/ocr/." . /s /e /y /d
.\.run_ce.bat
