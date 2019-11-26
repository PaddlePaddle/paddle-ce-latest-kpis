@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/deeplabv3+/." . /s /e /y
if exist data (rd /s /q data)
if exist model (rd /s /q model)
if not exist deeplabv3plus_gn_init (mklink /j  deeplabv3plus_gn_init  %data_path%\deeplabv3+\deeplabv3plus_gn_init)
if not exist deeplabv3plus_gn (mklink /j  deeplabv3plus_gn  %data_path%\deeplabv3+\deeplabv3plus_gn)
md model
md data
cd data
mklink /j  cityscape %data_path%\cityscape 
cd ..
.\.run_ce.bat
