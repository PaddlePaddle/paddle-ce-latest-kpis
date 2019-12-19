@echo off
rem This file is only used for continuous evaluation.
set CPU_NUM=1
python train.py --model_net Pix2pix --output output_pix2pix --net_G unet_256  --dataset cityscapes --train_list data/cityscapes/pix2pix_train_list --test_list data/cityscapes/pix2pix_test_list  --dropout False --gan_mode vanilla --batch_size 120 --epoch 1 --enable_ce --shuffle False --run_test False --save_checkpoints False --use_gpu False --print_freq 1 | python _ce.py
