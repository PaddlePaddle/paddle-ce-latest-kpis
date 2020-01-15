@echo off
set CUDA_VISIBLE_DEVICES=0
rem deeplabv3p
set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -i s/"BATCH_SIZE: 4"/"BATCH_SIZE: 2"/g configs/deeplabv3p_xception65_optic.yaml
%sed% -i s/"    NUM_EPOCHS: 10"/"    NUM_EPOCHS: 1"/g configs/deeplabv3p_xception65_optic.yaml
python pdseg/train.py --use_gpu --enable_ce --cfg ./configs/deeplabv3p_xception65_optic.yaml | grep "epoch"| gawk  -F "[= ]" "{if ($4 >=130) print \"kpis\tdeeplabv3p_loss_card1\t\"$8\"\nkpis\tdeeplabv3p_speed_card1\t\"$10}" | python _ce.py
rem icnet
%sed% -i s/"    NUM_EPOCHS: 10"/"    NUM_EPOCHS: 1"/g configs/icnet_optic.yaml
python pdseg/train.py --use_gpu --enable_ce --cfg ./configs/icnet_optic.yaml | grep "epoch"| gawk  -F "[= ]" "{if ($4 >=60) print \"kpis\ticnet_loss_card1\t\"$8\"\nkpis\ticnet_speed_card1\t\"$10}" | python _ce.py
rem unet
%sed% -i s/"    NUM_EPOCHS: 10"/"    NUM_EPOCHS: 1"/g configs/unet_optic.yaml
python pdseg/train.py --use_gpu --enable_ce --cfg ./configs/unet_optic.yaml | grep "epoch"| gawk  -F "[= ]" "{if ($4 >=60) print \"kpis\tunet_loss_card1\t\"$8\"\nkpis\tunet_speed_card1\t\"$10}" | python _ce.py
rem pspnet
%sed% -i s/"    NUM_EPOCHS: 10"/"    NUM_EPOCHS: 1"/g configs/pspnet_optic.yaml
python pdseg/train.py --use_gpu --enable_ce --cfg ./configs/pspnet_optic.yaml | grep "epoch"| gawk  -F "[= ]" "{if ($4 >=60) print \"kpis\tpspnet_loss_card1\t\"$8\"\nkpis\tpspnet_speed_card1\t\"$10}" | python _ce.py
rem hrnet
%sed% -i s/"    NUM_EPOCHS: 10"/"    NUM_EPOCHS: 1"/g configs/hrnet_optic.yaml
python pdseg/train.py --use_gpu --enable_ce --cfg ./configs/hrnet_optic.yaml | grep "epoch"| gawk  -F "[= ]" "{if ($4 >=60) print \"kpis\thrnet_loss_card1\t\"$8\"\nkpis\thrnet_speed_card1\t\"$10}" | python _ce.py


