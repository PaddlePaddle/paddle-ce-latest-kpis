#!/bin/bash
# This file is only used for continuous evaluation.
export CPU_NUM=1
python train.py --model_net Pix2pix --output output_pix2pix --net_G unet_256  --dataset cityscapes --train_list data/cityscapes/pix2pix_train_list --test_list data/cityscapes/pix2pix_test_list  --dropout False --gan_mode vanilla --batch_size 120 --epoch 1 --enable_ce --shuffle False --run_test False --save_checkpoints False --use_gpu False --print_freq 1 | python _ce.py
# infer
python infer.py --init_model output_pix2pix/checkpoints/0/ --image_size 256 --n_samples 10 --crop_size 256 --dataset_dir data/cityscapes/ --model_net Pix2pix --net_G unet_256 --test_list data/cityscapes/testB.txt --output ./infer_result/pix2pix/ --use_gpu false > $log_path/pix2pix_I.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/pix2pix_I.log ${log_path}/FAIL/pix2pix_I.log↩
        echo -e "pix2pix,infer,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/pix2pix_I.log ${log_path}/SUCCESS/pix2pix_I.log↩
        echo -e "pix2pix,infer,SUCCESS" >>${log_path}/result.log↩
fi
