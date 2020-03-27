@echo off
set models_dir=./../../parakeet_repo
rem copy models files
xcopy "%models_dir%/examples/transformer_tts/." . /s /e /y /d
if not exist data (mklink /j data %data_path%\DeepVoice3)
python -c "import nltk;nltk.download('punkt');nltk.download('cmudict')"
.\.run_ce.bat
