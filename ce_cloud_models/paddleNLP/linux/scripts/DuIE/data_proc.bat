@echo off
cd ../..

cd models_repo

cd examples\information_extraction\DuIE\

xcopy /y /c /h /r D:\ce_data\paddleNLP\DuIE\*  .\data\
