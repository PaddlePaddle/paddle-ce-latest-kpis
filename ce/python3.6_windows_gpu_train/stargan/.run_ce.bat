@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
rem train
python train.py --model_net StarGAN --output output_stargan --dataset celeba --crop_size 178 --image_size 128 --train_list ./data/celeba/list_attr_celeba.txt --test_list ./data/celeba/test_list_attr_celeba.txt  --gan_mode wgan --batch_size 1 --epoch 1 --run_test false --save_checkpoints true --shuffle false --use_gpu true --num_discriminator_time 1 --print_freq 1 --enable_ce | python _ce.py
rem infer
python infer.py --model_net StarGAN --init_model output_stargan/checkpoints/0/ --dataset_dir "data/celeba/" --image_size 128 --c_dim 5 --selected_attrs "Black_Hair,Blond_Hair,Brown_Hair,Male,Young" --use_gpu true > %log_path%/stargan_I.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\stargan_I.log  %log_path%\FAIL\stargan_I.log
        echo   stargan,infer,FAIL  >> %log_path%\result.log
        echo  infer of stargan failed!
) else (
        move  %log_path%\stargan_I.log  %log_path%\SUCCESS\stargan_I.log
        echo   stargan,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of stargan successfully!
)
