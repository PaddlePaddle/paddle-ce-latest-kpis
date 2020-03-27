@echo off
set CUDA_VISIBLE_DEVICES=0
python -u run_classifier.py  --task_name senta --use_cuda false --do_train true --do_val true --do_infer false --batch_size 16 --data_dir ./senta_data/ --vocab_path ./senta_data//word_dict.txt --checkpoints ./save_models --save_steps 500 --validation_steps 50 --epoch 1 --senta_config_path ./senta_config.json --skip_steps 10 --random_seed 0 --enable_ce True > sentiment_classification.log 2>&1
type sentiment_classification.log|grep "dev evaluation"|awk -F  "[:, ]" "END{print \"kpis\ttrain_loss_senta\t\"$6\"\nkpis\ttrain_acc_senta\t\"$11\"\nkpis\teach_step_duration_senta\t\"$16}"
