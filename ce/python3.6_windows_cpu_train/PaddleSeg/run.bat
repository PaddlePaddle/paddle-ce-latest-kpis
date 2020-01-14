@echo off
set seg_dir=./../../seg_repo
rem copy models files
xcopy "%seg_dir%/." . /s /e /y /d
rd /s /q dataset 
rd /s /q pretrained_model
mklink /j  dataset  %data_path%\PaddleSeg\dataset
mklink /j  pretrained_model  %data_path%\PaddleSeg\pretrained_model
rem pip install -r requirements.txt
pip install Cython
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
pip install tqdm
.\.run_ce.bat
