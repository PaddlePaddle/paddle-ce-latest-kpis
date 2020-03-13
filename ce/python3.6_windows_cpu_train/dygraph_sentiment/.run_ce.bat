@echo off
set FLAGS_cudnn_deterministic=True
set CUDA_VISIBLE_DEVICES=0
rem sentiment
python main.py --ce  --use_cuda false --model_type=bow_net --do_train=True --do_infer=True   --batch_size=256 --epoch 1 --random_seed 33 --validation_step 30 --save_steps 30 | python _ce.py

              







