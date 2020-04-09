@echo off
set models_dir=./../../models_repo
rem copy models files
xcopy "%models_dir%/dygraph/bmn/." . /s /e /y /d
if not exist fix_feat_100 (mklink /j fix_feat_100 %data_path%\bmn\fix_feat_100)
if not exist activitynet_1.3_annotations.json (mklink /h activitynet_1.3_annotations.json %data_path%\bmn\activitynet_1.3_annotations.json)

set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -i s!"../../PaddleCV/video/data/dataset/bmn/activitynet_1.3_annotations.json"!"activitynet_1.3_annotations.json"!g bmn.yaml

pip install pandas

.\.run_ce.bat
