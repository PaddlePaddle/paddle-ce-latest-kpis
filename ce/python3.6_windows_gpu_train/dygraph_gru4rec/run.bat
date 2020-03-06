@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/PaddleRec/gru4rec/dy_graph/." . /s /e /y /d

if not exist data (mklink /j data %data_path%\dygraph_gru4rec\data)

.\.run_ce.bat
