@echo off
set models_dir=./../../parakeet_repo
rem copy models files
xcopy "%models_dir%/examples/clarinet/." . /s /e /y /d
python -c "import nltk;nltk.download('punkt');nltk.download('cmudict')"
python -m pip install -U tqdm==4.15
if not exist data (mklink /j data %data_path%\Parakeet)

mklink /j "../wavenet" "%models_dir%\examples\wavenet"
call .run_ce.bat
rd /s /q "../wavenet"
