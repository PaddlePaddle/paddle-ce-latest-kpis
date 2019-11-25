@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
pip install -U paddlehub
python train.py --train_data ./data/train.tsv --test_data ./data/test.tsv --model_save_dir ./models --validation_steps 2 --save_steps 10 --print_steps 1 --batch_size 300 --epoch 1 --traindata_shuffle_buffer 20000 --word_emb_dim 128 --grnn_hidden_dim 128 --bigru_num 2  --base_learning_rate 1e-3  --emb_learning_rate 2  --crf_learning_rate 0.2 --word_dict_path ./conf/word.dic --label_dict_path ./conf/tag.dic --word_rep_dict_path ./conf/q2b.dic  --use_cuda false --enable_ce true | python _ce.py
				



