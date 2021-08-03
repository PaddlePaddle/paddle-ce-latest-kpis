@echo off
cd ../..

cd models_repo

if "%1"=="True" ( 
    python setup.py bdist_wheel

    for %%i in (".\dist\*.whl") do (
        set FileName=%%~nxi
    )

    python -m pip uninstall -y paddlenlp

    python -m pip install dist\%FileName%
)
cd examples\lexical_analysis\

python download.py --data_dir ./
