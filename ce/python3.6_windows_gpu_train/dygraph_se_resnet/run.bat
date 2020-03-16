@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/dygraph/se_resnet/." . /s /e /y /d
set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -i s/"batch_size = 64"/"batch_size = 32"/g train.py
.\.run_ce.bat
