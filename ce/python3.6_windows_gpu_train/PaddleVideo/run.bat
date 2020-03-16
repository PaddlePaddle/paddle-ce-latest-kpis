@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/video/." . /s /e /y /d
cd data/dataset
rd /s /q kinetics
mklink /j kinetics %data_path%\k400
rd /s /q bmn
mklink /j bmn %data_path%\bmn
rd /s /q youtube8m
mklink /j youtube8m %data_path%\youtube8m
cd ../..
pip install wget
pip install h5py
.\.run_ce.bat
