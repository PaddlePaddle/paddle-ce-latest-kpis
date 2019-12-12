@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/\image_classification/." . /s /e /y /d
cd data
if exist ILSVRC2012  (rd /s /q ILSVRC2012)
mklink /j  ILSVRC2012  %data_path%\ILSVRC2012
cd ..
mklink /j  ResNet50_pretrained  %data_path%\ILSVRC2012\ResNet50_pretrained
.\.run_ce.bat
