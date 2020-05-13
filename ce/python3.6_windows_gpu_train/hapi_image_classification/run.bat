@echo off
set models_dir=./../../hapi_repo
rem copy models files
xcopy "%models_dir%/examples/image_classification/." . /s /e /y /d
if not exist data (md data)
cd data
if exist ILSVRC2012  (rd /s /q ILSVRC2012)
mklink /j  ILSVRC2012  %data_path%\ILSVRC2012
cd ..
.\.run_ce.bat
