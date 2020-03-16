@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set CUDA_VISIBLE_DEVICES=0
rem train
python run_classifier.py --task_name COLA --use_cuda true --do_train true --do_test true --batch_size 64 --init_pretraining_params "./data/pretrained_models/chinese_L-12_H-768_A-12\params/" --data_dir "./data/glue_data/CoLA/" --vocab_path "./data/pretrained_models/chinese_L-12_H-768_A-12/vocab.txt " --checkpoints "./data/saved_model/cola_models" --save_steps 1000 --weight_decay  0.01 --warmup_proportion 0.1 --validation_steps 100 --epoch 1 --max_seq_len 128 --bert_config_path "./data/pretrained_models/chinese_L-12_H-768_A-12/bert_config.json" --learning_rate 5e-5  --skip_steps 10  --shuffle false --enable_ce | python _ce.py

