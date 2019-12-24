@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/human_pose_estimation/." . /s /e /y /d

pip install pathlib
pip install --upgrade numpy
if exist pretrained   (rd /s /q pretrained)
md pretrained
cd pretrained
mklink /j  resnet_50 %data_path%\human_pose\resnet_50
mklink /j  pose-resnet50-mpii-384x384 %data_path%\human_pose\pose-resnet50-mpii-384x384
mklink /j  pose-resnet50-coco-384x288 %data_path%\human_pose\pose-resnet50-coco-384x288
cd ..
if exist data (rd /s /q data)
md data
cd data
mklink /j  mpii %data_path%\MPII
rem coco
mklink /j  coco %data_path%\human_pose_coco
cd ..

.\.run_ce.bat
