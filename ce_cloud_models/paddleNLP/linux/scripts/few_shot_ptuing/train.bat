@echo off
cd ../..

md log\few_shot_ptuing

set logpath=%cd%\log\few_shot_ptuing

cd models_repo\examples\few_shot\p-tuning\

python -u -m paddle.distributed.launch --gpus %2 ptuning.py --task_name %3 --device %1 --p_embedding_num 1 --save_dir "checkpoints/%3" --batch_size 16 --learning_rate 5E-5 --epochs 1 --save_steps 10 --max_seq_length 512 > %logpath%/train_%3_%1.log 2>&1
