#!/bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.5
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
pip install -U paddlehub
python train.py --train_data ./data/train.tsv --test_data ./data/test.tsv --model_save_dir ./models --validation_steps 2 --save_steps 10 --batch_size 100 --epoch 2 --use_cuda false --traindata_shuffle_buffer 200000 --word_emb_dim 768 --grnn_hidden_dim 768 --bigru_num 2  --base_learning_rate 1e-3 --emb_learning_rate 5 --crf_learning_rate 0.2 --word_dict_path ./conf/word.dic --label_dict_path ./conf/tag.dic  --word_rep_dict_path ./conf/q2b.dic --enable_ce true --random_seed  90 | python _ce.py
# eval
python eval.py --batch_size 200 --word_emb_dim 128 --grnn_hidden_dim 128 --bigru_num 2 --use_cuda False --init_checkpoint ./model_baseline --test_data ./data/test.tsv --word_dict_path ./conf/word.dic --label_dict_path ./conf/tag.dic --word_rep_dict_path ./conf/q2b.dic > $log_path/lac_eval.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/lac_eval.log ${log_path}/FAIL/lac_eval.log↩
        echo -e "lac,eval,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/lac_eval.log ${log_path}/SUCCESS/lac_eval.log↩
        echo -e "lac,eval,SUCCESS" >>${log_path}/result.log↩
fi
# infer
python predict.py --batch_size 200  --word_emb_dim 128 --grnn_hidden_dim 128 --bigru_num 2 --use_cuda False --init_checkpoint ./model_baseline --infer_data ./data/infer.tsv --word_dict_path ./conf/word.dic  --label_dict_path ./conf/tag.dic --word_rep_dict_path ./conf/q2b.dic > $log_path/lac_infer.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/lac_infer.log ${log_path}/FAIL/lac_infer.log↩
        echo -e "lac,infer,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/lac_infer.log ${log_path}/SUCCESS/lac_infer.log↩
        echo -e "lac,infer,SUCCESS" >>${log_path}/result.log↩
fi
# inference
python inference_model.py --word_emb_dim 128 --grnn_hidden_dim 128 --bigru_num 2  --use_cuda false  --init_checkpoint ./model_baseline --word_dict_path ./conf/word.dic --label_dict_path ./conf/tag.dic  --word_rep_dict_path ./conf/q2b.dic --inference_save_dir ./infer_model > $log_path/lac_infer_save.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/lac_infer_save.log ${log_path}/FAIL/lac_infer_save.log↩
        echo -e "lac,infer_save,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/lac_infer_save.log ${log_path}/SUCCESS/lac_infer_save.log↩
        echo -e "lac,infer_save,SUCCESS" >>${log_path}/result.log↩
fi
# lexical_analysis_ernie
# train
python run_ernie_sequence_labeling.py --mode train --ernie_config_path ./pretrained/ernie_config.json --model_save_dir ./ernie_models --init_pretraining_params ./pretrained/params/ --epoch 1 --save_steps 334 --validation_steps 334 --base_learning_rate 2e-4 --crf_learning_rate 0.2 --init_bound 0.1 --print_steps 1 --vocab_path ./pretrained/vocab.txt --batch_size 12 --random_seed 0 --num_labels 57  --max_seq_len 128 --train_data ./data/train.tsv --test_data ./data/test.tsv  --label_map_config ./conf/label_map.json --do_lower_case true --use_cuda false > $log_path/lac_ernie_train.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/lac_ernie_train.log ${log_path}/FAIL/lac_ernie_train.log↩
        echo -e "lac_ernie,train,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/lac_ernie_train.log ${log_path}/SUCCESS/lac_ernie_train.log↩
        echo -e "lac_ernie,train,SUCCESS" >>${log_path}/result.log↩
fi
# eval
python run_ernie_sequence_labeling.py --mode eval --ernie_config_path ./pretrained/ernie_config.json --init_checkpoint ./model_finetuned --init_bound 0.1 --vocab_path ./pretrained/vocab.txt --batch_size 64 --random_seed 0 --num_labels 57 --max_seq_len 128 --test_data ./data/test.tsv --label_map_config ./conf/label_map.json  --do_lower_case true  --use_cuda false > $log_path/lac_ernie_eval.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/lac_ernie_eval.log ${log_path}/FAIL/lac_ernie_eval.log↩
        echo -e "lac_ernie,eval,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/lac_ernie_eval.log ${log_path}/SUCCESS/lac_ernie_eval.log↩
        echo -e "lac_ernie,eval,SUCCESS" >>${log_path}/result.log↩
fi
# infer
python run_ernie_sequence_labeling.py --mode infer --ernie_config_path ./pretrained/ernie_config.json --init_checkpoint ./model_finetuned --init_bound 0.1 --vocab_path ./pretrained/vocab.txt --batch_size 64 --random_seed 0 --num_labels 57 --max_seq_len 128 --test_data ./data/test.tsv --label_map_config ./conf/label_map.json  --do_lower_case true  --use_cuda false > $log_path/lac_ernie_infer.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/lac_ernie_infer.log ${log_path}/FAIL/lac_ernie_infer.log↩
        echo -e "lac_ernie,infer,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/lac_ernie_infer.log ${log_path}/SUCCESS/lac_ernie_infer.log↩
        echo -e "lac_ernie,infer,SUCCESS" >>${log_path}/result.log↩
fi




