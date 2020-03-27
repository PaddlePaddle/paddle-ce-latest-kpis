#!/bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95

python run_classifier.py --task_name simnet --use_cuda false --do_train True --do_valid True --do_test False --do_infer False --batch_size 128 --train_data_dir ./data/zhidao --valid_data_dir ./data/zhidao --test_data_dir ./data/zhidao --infer_data_dir ./data/zhidao --output_dir ./model_files --config_path  ./config/bow_pairwise.json --vocab_path ./data/term2id.dict --epoch 6 --save_steps 1000 --validation_steps 1 --compute_accuracy True --lamda 0.958 --task_mode pairwise --init_checkpoint ""  --enable_ce | python _ce.py
# eval
python run_classifier.py --task_name simnet --use_cuda false --do_test True --verbose_result True --batch_size 128 --test_data_dir ./data/test_pairwise_data --test_result_path ./test_result --config_path ./config/bow_pairwise.json --vocab_path ./data/term2id.dict --task_mode pairwise --compute_accuracy False --lamda 0.95 --init_checkpoint ./model_files/simnet_bow_pairwise_pretrained_model/ > $log_path/simnet_E.log 2>&1
if [ $? -ne 0 ]then
        mv ${log_path}/simnet_E.log ${log_path}/FAIL/simnet_E.log
		echo -e "\033[33m eval of simnet failed! \033[0m"
        echo -e "simnet,eval,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/simnet_E.log ${log_path}/SUCCESS/simnet_E.log
		echo -e "\033[33m eval of simnet successfully! \033[0m"
        echo -e "simnet,eval,SUCCESS" >>${log_path}/result.log
fi
# infer
python run_classifier.py --task_name simnet --use_cuda false --do_infer True --batch_size 128 --infer_data_dir ./data/infer_data --infer_result_path ./infer_result --config_path ./config/bow_pairwise.json --vocab_path ./data/term2id.dict --task_mode pairwise --init_checkpoint ./model_files/simnet_bow_pairwise_pretrained_model/ > $log_path/simnet_I.log 2>&1
if [ $? -ne 0 ]then
        mv ${log_path}/simnet_I.log ${log_path}/FAIL/simnet_I.log
		echo -e "\033[33m infer of simnet failed! \033[0m"
        echo -e "simnet,infer,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/simnet_I.log ${log_path}/SUCCESS/simnet_I.log
		echo -e "\033[33m infer of simnet successfully! \033[0m"
        echo -e "simnet,infer,SUCCESS" >>${log_path}/result.log
fi
