@echo off
set ce_mode=1
python train.py --model=crnn_ctc --total_step=100 --save_model_period=100 --eval_period=100 --save_model_dir=output_ctc --use_gpu=False | python _ce.py
