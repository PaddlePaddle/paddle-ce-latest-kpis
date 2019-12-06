@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/PaddleVideo/." . /s /e /y /d
cd data/dataset
rd /s /q kinetics
mklink /j kinetics %data_path%\k400
cd ../..
pip install wget
pip install h5py
.\.run_ce.bat
