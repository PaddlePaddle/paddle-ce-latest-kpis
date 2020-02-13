@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/yolov3/." . /s /e /y /d
rd /s /q weights
mklink /j weights  %data_path%\yolov3\weights
rd /s /q dataset
mklink /j dataset  %data_path%\yolov3\dataset
.\.run_ce.bat
