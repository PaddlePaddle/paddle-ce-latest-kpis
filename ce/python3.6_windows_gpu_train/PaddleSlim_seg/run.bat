@echo off
set models_dir=./../../seg_repo
rem copy models files
xcopy "%models_dir%/." . /s /e /y /d
pip install -U paddleslim
cd dataset
if  not exist cityscapes (mklink /j cityscapes %data_path%\cityscape)
cd ..
if exist pretrained_model (rd /s /q pretrained_model)
mklink /j pretrained_model %data_path%\PaddleSlim_seg\pretrained_model

.\.run_ce.bat