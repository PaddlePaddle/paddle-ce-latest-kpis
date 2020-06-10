@echo off
set models_dir=./../../parakeet_repo
rem copy models files
xcopy "%models_dir%/examples/deepvoice3/." . /s /e /y /d
if not exist data (mklink /j data %data_path%\Parakeet)
python -c "import nltk;nltk.download('punkt');nltk.download('cmudict')"
python -m pip install -U tqdm==4.15
.\.run_ce.bat
