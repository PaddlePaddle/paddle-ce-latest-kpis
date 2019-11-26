@echo off
set CUDA_VISIBLE_DEVICES=0
python train_elem.py --model=ResNet50 --loss_name arcmargin --total_iter_num=10 --test_iter_step=10 --model_save_dir=output --save_iter_step=10 --train_batch_size 16 --test_batch_siz 16 --use_gpu=True --enable_ce true | python _ce.py 