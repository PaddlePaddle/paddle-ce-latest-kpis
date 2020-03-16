@echo off
set FLAGS_enable_parallel_graph=1
set FLAGS_sync_nccl_allreduce=1
set FLAGS_fraction_of_gpu_memory_to_use=0.95
set CUDA_VISIBLE_DEVICES=0
set ce_mode=1
rem train
python run_classifier.py --use_cuda true --do_train true --do_val true --epoch 50 --lr 0.002 --batch_size 64 --save_checkpoint_dir ./save_models/textcnn --save_steps 2000 --validation_steps 100 --skip_steps 10 > %log_path%/emotion_detection_T.log 2>&1 
type %log_path%\emotion_detection_T.log|grep "dev evaluation" |awk -F "[:,]" "{print $4}"|awk "NR==1{max=$1;next}{max=max>$1?max:$1}END{print  \"kpis\tdev_acc\t\"max}"|python _ce.py
if not %errorlevel% == 0 (
        move  %log_path%\emotion_detection_T.log  %log_path%\FAIL\emotion_detection_T.log
        echo   emotion_detection,train,FAIL  >> %log_path%\result.log
        echo   train of emotion_detection failed!
) else (
        move  %log_path%\emotion_detection_T.log  %log_path%\SUCCESS\emotion_detection_T.log
        echo   emotion_detection,train,SUCCESS  >> %log_path%\result.log
        echo   train of emotion_detection successfully!
)