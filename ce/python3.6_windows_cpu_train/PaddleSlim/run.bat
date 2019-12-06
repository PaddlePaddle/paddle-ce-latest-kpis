@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleSlim/." . /s /e /y /d
if not exist data (
md data
mklink /j data\ILSVRC2012 %data_path%\ILSVRC2012
)
if not exist pretrain (mklink /j pretrain %data_path%\PaddleSlim)
if exist checkpoints (rd /s /q checkpoints)
.\.run_ce.bat
