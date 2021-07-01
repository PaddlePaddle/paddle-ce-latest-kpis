@echo off
cd ../..

md log\language_model_ernie
set logpath=%cd%\log\language_model_ernie

cd models_repo\examples\language_model\ernie\

md output_dir
md output_dir\log

python -m paddle.distributed.fleet.launch --gpus %2 --log_dir ./output_dir/log run_pretraining.py --global_bsz 64 --micro_bsz 1 --max_seq_len 512 --ernie_config_file config/ernie_base_config.json --learning_rate 1e-4 --log_steps 1 --num_train_steps 100 --save_steps 10 --output_dir ./output_dir/ --use_recompute true --use_sharding true --use_sop false --num_mp=1 --num_sharding=1 --num_pp=1 --num_dp=1 > %logpath%\train_%1.log 2>&1

