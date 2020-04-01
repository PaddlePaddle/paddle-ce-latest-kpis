#!/bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export ce_mode=1
python run_classifier.py --use_cuda false --do_train true --do_val true --epoch 10 --lr 0.002 --batch_size 64 --save_checkpoint_dir ./save_models/textcnn --save_steps 200 --validation_steps 200 --skip_steps 200 --random_seed 90 --enable_ce true | python _ce.py
# eval
python run_classifier.py --use_cuda false --do_val true --batch_size 128  --init_checkpoint ./save_models/textcnn/step_1510 > $log_path/emotion_detection_E.log 2>&1
if [ $? -ne 0 ]; then
        mv ${log_path}/emotion_detection_E.log ${log_path}/FAIL/emotion_detection_E.log
		echo -e "\033[33m eval of emotion_detection failed! \033[0m"
        echo -e "emotion_detection,eval,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/emotion_detection_E.log ${log_path}/SUCCESS/emotion_detection_E.log
		echo -e "\033[33m eval of emotion_detection successfully! \033[0m"
        echo -e "emotion_detection,eval,SUCCESS" >>${log_path}/result.log
fi
#infer
run_classifier.py --use_cuda false --do_infer true --batch_size 32 --init_checkpoint ./save_models/textcnn/step_1510  > $log_path/emotion_detection_I.log 2>&1
if [ $? -ne 0 ]; then
        mv ${log_path}/emotion_detection_I.log ${log_path}/FAIL/emotion_detection_I.log
		echo -e "\033[33m infer of emotion_detection failed! \033[0m"
        echo -e "emotion_detection,infer,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/emotion_detection_I.log ${log_path}/SUCCESS/emotion_detection_I.log
		echo -e "\033[33m infer of emotion_detection successfully! \033[0m"
        echo -e "emotion_detection,infer,SUCCESS" >>${log_path}/result.log
fi

# save_inference_model
python inference_model.py --use_cuda false --do_save_inference_model true --init_checkpoint ./save_models/textcnn/step_1510 --inference_model_dir ./inference_model > $log_path/emotion_detection_save_infer.log 2>&1
if [ $? -ne 0 ]; then
        mv ${log_path}/emotion_detection_save_infer.log ${log_path}/FAIL/emotion_detection_save_infer.log
		echo -e "\033[33m save_infer of emotion_detection failed! \033[0m"
        echo -e "emotion_detection,save_infer,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/emotion_detection_save_infer.log ${log_path}/SUCCESS/emotion_detection_save_infer.log
		echo -e "\033[33m save_infer of emotion_detection successfully! \033[0m"
        echo -e "emotion_detection,save_infer,SUCCESS" >>${log_path}/result.log
fi
