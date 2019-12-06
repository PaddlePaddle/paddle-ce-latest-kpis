@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/rcnn/." . /s /e /y /d
cd dataset
if exist coco (rd /s /q coco)
mklink /j coco %data_path%\COCO17
cd ..
if not exist imagenet_resnet50_fusebn (mklink /j imagenet_resnet50_fusebn  %data_path%\rcnn\imagenet_resnet50_fusebn)
.\.run_ce.bat
