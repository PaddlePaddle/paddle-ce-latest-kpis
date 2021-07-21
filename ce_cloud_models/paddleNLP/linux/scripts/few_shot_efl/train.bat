@echo off
cd ../..

md log\few_shot_efl

set logpath=%cd%\log\few_shot_efl

cd models_repo\examples\few_shot\efl\

python -u -m paddle.distributed.launch --gpus %2 train.py --task_name %3 --device %1 --negative_num 1 --save_dir "checkpoints/%3" --batch_size 16 --learning_rate 5E-5 --epochs 1 --max_seq_length 512 --save_steps 10 > %logpath%/train_%3_%1.log 2>&1
