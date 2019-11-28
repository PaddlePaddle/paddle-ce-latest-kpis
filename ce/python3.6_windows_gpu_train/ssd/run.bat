@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/ssd/." . /s /e /y /d
cd data
if exist pascalvoc  (rd /s /q pascalvoc)
mklink /j pascalvoc %data_path%\pascalvoc
cd ..
cd pretrained
if not exist ssd_mobilenet_v1_coco (mklink /j  ssd_mobilenet_v1_coco %data_path%\ssd_mobilenet_v1_coco)
cd ..
.\.run_ce.bat
