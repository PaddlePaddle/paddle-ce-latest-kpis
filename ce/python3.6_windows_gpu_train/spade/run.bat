@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleCV/PaddleGAN/." . /s /e /y /d
if exist data (rd /s /q data)
if exist VGG19_pretrained (rd /s /q VGG19_pretrained)
mklink /j data %data_path%\SPADE
mklink /j VGG19_pretrained %data_path%\SPADE\VGG19_pretrained

.\.run_ce.bat
