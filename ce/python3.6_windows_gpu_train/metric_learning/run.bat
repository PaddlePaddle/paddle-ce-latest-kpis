@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/metric_learning/." . /s /e /y /d
cd data
if exist Stanford_Online_Products (rd /s /q Stanford_Online_Products)
mklink /j Stanford_Online_Products  %data_path%\Stanford_Online_Products
cd ..
.\.run_ce.bat
