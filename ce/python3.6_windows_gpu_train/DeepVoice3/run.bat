@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleSpeech/DeepVoice3/." . /s /e /y /d
if not exist data (mklink /j data %data_path%\DeepVoice3)
pip install -r requirements.txt
python -c "import nltk;nltk.download('punkt');nltk.download('cmudict')"
.\.run_ce.bat
