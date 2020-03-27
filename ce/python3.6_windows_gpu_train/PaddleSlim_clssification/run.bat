@echo off
set slim_dir=./../../slim_repo
rem copy models files
xcopy "%slim_dir%/." . /s /e /y /d
pip install -U paddleslim
cd demo
if not exist data (
md data
mklink /j data\ILSVRC2012 %data_path%\ILSVRC2012
)
cd ..
echo f | xcopy  _ce.py  "demo/distillation/_ce.py" /y
echo f | xcopy  _ce.py  "demo/quant/quant_aware/_ce.py" /y 
echo f | xcopy  _ce.py  "demo/prune/_ce.py" /y
echo f | xcopy  _ce.py  "demo/nas/_ce.py" /y 


.\.run_ce.bat