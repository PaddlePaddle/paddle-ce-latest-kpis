@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/face_detection/." . /s /e /y /d
if exist data (rd /s /q data)
mklink /j  data %data_path%\face_detection\WIDERFACE 
if not exist vgg_ilsvrc_16_fc_reduced  (mklink /j vgg_ilsvrc_16_fc_reduced %data_path%\face_detection\vgg_ilsvrc_16_fc_reduced)
if not exist PyramidBox_WiderFace  (mklink /j PyramidBox_WiderFace %data_path%\face_detection\PyramidBox_WiderFace)
.\.run_ce.bat
