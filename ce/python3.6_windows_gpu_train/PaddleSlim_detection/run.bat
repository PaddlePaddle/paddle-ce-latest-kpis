@echo off
set models_dir=./../../detection_repo
rem copy models files
xcopy "%models_dir%/." . /s /e /y /d
set PYTHONPATH=.;%PYTHONPATH%
cd dataset
if exist coco (rd /s /q coco)
if exist voc (rd /s /q voc)
mklink /j coco  %data_path%\COCO17
mklink /j voc  %data_path%\pascalvoc
cd ..
pip install Cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip install tqdm
.\.run_ce.bat
