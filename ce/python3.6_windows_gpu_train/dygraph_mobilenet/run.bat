@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/dygraph/mobilenet/." . /s /e /y /d
if not exist data ( 
md data
cd data
mklink /j ILSVRC2012 %data_path%\ILSVRC2012
cd ..
)

.\.run_ce.bat
