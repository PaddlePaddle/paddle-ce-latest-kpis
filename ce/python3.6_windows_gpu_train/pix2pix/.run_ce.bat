@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
rem train
python train.py --model_net Pix2pix --output output_pix2pix --net_G unet_256  --dataset cityscapes --train_list data/cityscapes/pix2pix_train_list --test_list data/cityscapes/pix2pix_test_list  --dropout False --gan_mode vanilla --batch_size 64 --epoch 1 --enable_ce --shuffle False --run_test False --save_checkpoints True --use_gpu True  | python _ce.py
rem infer
python infer.py --init_model output_pix2pix/checkpoints/0/ --image_size 256 --n_samples 10 --crop_size 256 --dataset_dir data/cityscapes/ --model_net Pix2pix --net_G unet_256 --test_list data/cityscapes/testB.txt --output ./infer_result/pix2pix/ --use_gpu true > %log_path%/pix2pix_I.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\pix2pix_I.log  %log_path%\FAIL\pix2pix_I.log
        echo   pix2pix,infer,FAIL  >> %log_path%\result.log
        echo  infer of pix2pix failed!
) else (
        move  %log_path%\pix2pix_I.log  %log_path%\SUCCESS\pix2pix_I.log
        echo   pix2pix,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of pix2pix successfully!
)
