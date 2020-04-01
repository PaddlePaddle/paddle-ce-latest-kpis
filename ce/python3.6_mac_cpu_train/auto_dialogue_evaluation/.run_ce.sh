#!/bin/bash
export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1.0
export ce_mode=1 
python -u main.py --do_train=true --use_cuda=false --loss_type="CLS"  --max_seq_len=50  --save_model_path=output_train  --save_param="params"  --training_file="data/input/data/unlabel_data/train.ids" --epoch=20 --print_step=1 --save_step=400 --batch_size=256  --hidden_size=256 --emb_size=256  --vocab_size=484016 --learning_rate=0.001 --sample_pro=0.1 --enable_ce="store_true" | python _ce.py
#infer
python -u main.py --do_predict=true --use_cuda=false --predict_file="data/input/data/unlabel_data/test.ids" --init_from_params="output_train/params" --loss_type="CLS" --output_prediction_file="output_infer" --max_seq_len=50 --batch_size=256 --hidden_size=256  --emb_size=256 --vocab_size=484016 > $log_path/dialogue_evaluation_I.log 2>&1
if [ $? -ne 0 ]; then
        mv ${log_path}/dialogue_evaluation_I.log ${log_path}/FAIL/dialogue_evaluation_I.log
		echo -e "\033[33m infer of dialogue_evaluation failed! \033[0m"
        echo -e "dialogue_evaluation,infer,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/dialogue_evaluation_I.log ${log_path}/SUCCESS/dialogue_evaluation_I.log
		echo -e "\033[33m infer of dialogue_evaluation successfully! \033[0m"
        echo -e "dialogue_evaluation,infer,SUCCESS" >>${log_path}/result.log
fi




