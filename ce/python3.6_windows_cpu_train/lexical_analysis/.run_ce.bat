@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
pip install -U paddlehub
python train.py --train_data ./data/train.tsv --test_data ./data/test.tsv --model_save_dir ./models --validation_steps 2 --save_steps 10 --batch_size 100 --epoch 2 --use_cuda false --traindata_shuffle_buffer 200000 --word_emb_dim 768 --grnn_hidden_dim 768 --bigru_num 2  --base_learning_rate 1e-3 --emb_learning_rate 5 --crf_learning_rate 0.2 --word_dict_path ./conf/word.dic --label_dict_path ./conf/tag.dic  --word_rep_dict_path ./conf/q2b.dic --enable_ce true --random_seed  90 | python _ce.py				



