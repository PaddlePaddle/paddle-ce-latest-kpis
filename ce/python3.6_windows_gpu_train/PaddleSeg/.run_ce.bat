@echo off
set CUDA_VISIBLE_DEVICES=0
rem deeplabv3p
set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -i s/"BATCH_SIZE: 4"/"BATCH_SIZE: 2"/g configs/deeplabv3p_xception65_optic.yaml
%sed% -i s/"    NUM_EPOCHS: 10"/"    NUM_EPOCHS: 1"/g configs/deeplabv3p_xception65_optic.yaml
python pdseg/train.py --use_gpu --enable_ce --cfg ./configs/deeplabv3p_xception65_optic.yaml | grep "epoch"| gawk  -F "[= ]" "{if ($4 >=130) print \"kpis\tdeeplabv3p_loss_card1\t\"$8\"\nkpis\tdeeplabv3p_speed_card1\t\"$10}" | python _ce.py
rem eval
python pdseg/eval.py --use_gpu --cfg ./configs/deeplabv3p_xception65_optic.yaml > %log_path%/deeplabv3p_xception65_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\deeplabv3p_xception65_E.log  %log_path%\FAIL\deeplabv3p_xception65_E.log
        echo   deeplabv3p_xception65,eval,FAIL  >> %log_path%\result.log
        echo   eval of deeplabv3p_xception65 failed!
) else (
        move  %log_path%\deeplabv3p_xception65_E.log  %log_path%\SUCCESS\deeplabv3p_xception65_E.log
        echo   deeplabv3p_xception65,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of deeplabv3p_xception65 successfully!
)

rem icnet
%sed% -i s/"    NUM_EPOCHS: 10"/"    NUM_EPOCHS: 1"/g configs/icnet_optic.yaml
python pdseg/train.py --use_gpu --enable_ce --cfg ./configs/icnet_optic.yaml | grep "epoch"| gawk  -F "[= ]" "{if ($4 >=60) print \"kpis\ticnet_loss_card1\t\"$8\"\nkpis\ticnet_speed_card1\t\"$10}" | python _ce.py
rem eval
python pdseg/eval.py --use_gpu --cfg ./configs/icnet_optic.yaml > %log_path%/icnet_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\icnet_E.log  %log_path%\FAIL\icnet_E.log
        echo   icnet,eval,FAIL  >> %log_path%\result.log
        echo   eval of icnet failed!
) else (
        move  %log_path%\icnet_E.log  %log_path%\SUCCESS\icnet_E.log
        echo   icnet,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of icnet successfully!
)


rem unet
%sed% -i s/"    NUM_EPOCHS: 10"/"    NUM_EPOCHS: 1"/g configs/unet_optic.yaml
python pdseg/train.py --use_gpu --enable_ce --cfg ./configs/unet_optic.yaml | grep "epoch"| gawk  -F "[= ]" "{if ($4 >=60) print \"kpis\tunet_loss_card1\t\"$8\"\nkpis\tunet_speed_card1\t\"$10}" | python _ce.py
rem eval
python pdseg/eval.py --use_gpu --cfg ./configs/unet_optic.yaml > %log_path%/unet_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\unet_E.log  %log_path%\FAIL\unet_E.log
        echo   unet,eval,FAIL  >> %log_path%\result.log
        echo   eval of unet failed!
) else (
        move  %log_path%\unet_E.log  %log_path%\SUCCESS\unet_E.log
        echo   unet,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of unet successfully!
)

rem pspnet
%sed% -i s/"    NUM_EPOCHS: 10"/"    NUM_EPOCHS: 1"/g configs/pspnet_optic.yaml
python pdseg/train.py --use_gpu --enable_ce --cfg ./configs/pspnet_optic.yaml | grep "epoch"| gawk  -F "[= ]" "{if ($4 >=60) print \"kpis\tpspnet_loss_card1\t\"$8\"\nkpis\tpspnet_speed_card1\t\"$10}" | python _ce.py
rem eval
python pdseg/eval.py --use_gpu --cfg ./configs/unet_optic.yaml > %log_path%/pspnet_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\pspnet_E.log  %log_path%\FAIL\pspnet_E.log
        echo   pspnet,eval,FAIL  >> %log_path%\result.log
        echo   eval of pspnet failed!
) else (
        move  %log_path%\pspnet_E.log  %log_path%\SUCCESS\pspnet_E.log
        echo   pspnet,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of pspnet successfully!
)

rem hrnet
%sed% -i s/"    NUM_EPOCHS: 10"/"    NUM_EPOCHS: 1"/g configs/hrnet_optic.yaml
python pdseg/train.py --use_gpu --enable_ce --cfg ./configs/hrnet_optic.yaml | grep "epoch"| gawk  -F "[= ]" "{if ($4 >=60) print \"kpis\thrnet_loss_card1\t\"$8\"\nkpis\thrnet_speed_card1\t\"$10}" | python _ce.py
rem eval
python pdseg/eval.py --use_gpu --cfg ./configs/hrnet_optic.yaml > %log_path%/unet_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\hrnet_E.log  %log_path%\FAIL\hrnet_E.log
        echo   hrnet,eval,FAIL  >> %log_path%\result.log
        echo   eval of hrnet failed!
) else (
        move  %log_path%\unet_E.log  %log_path%\SUCCESS\hrnet_E.log
        echo   hrnet,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of hrnet successfully!
)

