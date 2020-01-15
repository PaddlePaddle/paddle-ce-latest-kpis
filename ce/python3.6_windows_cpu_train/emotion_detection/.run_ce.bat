@echo off
set FLAGS_enable_parallel_graph=1
set FLAGS_sync_nccl_allreduce=1
set FLAGS_fraction_of_gpu_memory_to_use=0.95
set ce_mode=1
rem train
python run_classifier.py --use_cuda false --do_train true --do_val true --epoch 10 --lr 0.002 --batch_size 64 --save_checkpoint_dir ./save_models/textcnn --save_steps 200 --validation_steps 200 --skip_steps 200 --random_seed 90 --enable_ce true | python _ce.py

rem eval
python run_classifier.py --use_cuda false --do_val true --batch_size 128  --init_checkpoint ./save_models/textcnn/step_756 > %log_path%/emotion_detection_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\emotion_detection_E.log  %log_path%\FAIL\emotion_detection_E.log
        echo   emotion_detection,eval,FAIL  >> %log_path%\result.log
        echo   eval of emotion_detection failed!
) else (
        move  %log_path%\emotion_detection_E.log  %log_path%\SUCCESS\emotion_detection_E.log
        echo   emotion_detection,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of emotion_detection successfully!
 )
rem infer
python run_classifier.py  --do_infer false --batch_size 32 --init_checkpoint ./save_models/textcnn/step_756  > %log_path%/emotion_detection_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\emotion_detection_I.log  %log_path%\FAIL\emotion_detection_I.log
        echo   emotion_detection,infer,FAIL  >> %log_path%\result.log
        echo   infer of emotion_detection failed!
) else (
        move  %log_path%\emotion_detection_I.log  %log_path%\SUCCESS\emotion_detection_I.log
        echo   emotion_detection,infer,SUCCISS  >> %log_path%\result.log
        echo   infer of emotion_detection successfully!
 )
rem save_inference_model
python inference_model.py --use_cuda false --do_save_inference_model true --init_checkpoint ./save_models/textcnn/step_756 --inference_model_dir ./inference_model > %log_path%/emotion_detection_save_infer.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\emotion_detection_save_infer.log  %log_path%\FAIL\emotion_detection_save_infer.log
        echo   emotion_detection,save_infer,FAIL  >> %log_path%\result.log
        echo   save_infer of emotion_detection failed!
) else (
        move  %log_path%\emotion_detection_save_infer.log  %log_path%\SUCCESS\emotion_detection_save_infer.log
        echo   emotion_detection,save_infer,SUCCISS  >> %log_path%\result.log
        echo   save_infer of emotion_detection successfully!
 )				