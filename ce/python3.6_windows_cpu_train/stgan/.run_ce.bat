@echo off
rem This file is only used for continuous evaluation.
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0
rem train
python train.py --model_net STGAN --output output_stgan --dataset celeba --crop_size 170 --image_size 128 --train_list ./data/celeba/list_attr_celeba.txt --test_list ./data/celeba/test_list_attr_celeba.txt  --gan_mode wgan  --batch_size 32 --print_freq 1 --num_discriminator_time 5 --epoch 1 --run_test false --save_checkpoints true --shuffle false --use_gpu false --num_discriminator_time 1 --enable_ce | python _ce.py
rem infer
python infer.py --model_net STGAN  --init_model output_stgan/checkpoints/0/ --dataset_dir "data/celeba/" --image_size 128 --use_gpu false > %log_path%/stgan_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\stgan_I.log  %log_path%\FAIL\stgan_I.log
        echo   stgan,infer,FAIL  >> %log_path%\result.log
        echo  infer of stgan failed!
) else (
        move  %log_path%\stgan_I.log  %log_path%\SUCCESS\stgan_I.log
        echo   stgan,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of stgan successfully!
 )
