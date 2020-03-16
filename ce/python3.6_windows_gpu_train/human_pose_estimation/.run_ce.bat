@echo off
set CUDA_VISIBLE_DEVICES=0
set ce_mode=1
python train.py --batch_size 8 --dataset coco --num_epochs=1  --enable_ce true --use_gpu true | python _ce.py >  %log_path%/humanpose_T.log