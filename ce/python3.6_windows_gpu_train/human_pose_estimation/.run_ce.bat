@echo off
set CUDA_VISIBLE_DEVICES=0
set ce_mode=1
python train.py --batch_size 8 --dataset coco --num_epochs=1  --enable_ce true --use_gpu true | python _ce.py >  %log_path%/humanpose_T.log
rem eval
python val.py --batch_size 8 --dataset coco --checkpoint pretrained/pose-resnet50-coco-384x288 --data_root data/coco >%log_path%/human_pose_coco_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\human_pose_coco_E.log  %log_path%\FAIL\human_pose_coco_E.log
        echo   human_pose_coco,eval,FAIL  >> %log_path%\result.log
        echo   evaling of human_pose_coco failed!
) else (
        move  %log_path%\human_pose_coco_E.log  %log_path%\SUCCESS\human_pose_coco_E.log
        echo   human_pose_coco,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of human_pose_coco successfully!
)
rem infer
python test.py --dataset coco --checkpoint pretrained/pose-resnet50-coco-384x288 > %log_path%/human_pose_coco_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\human_pose_coco_I.log  %log_path%\FAIL\human_pose_coco_I.log
        echo   human_pose_coco,infer,FAIL  >> %log_path%\result.log
        echo   infering of human_pose_coco failed!
) else (
        move  %log_path%\human_pose_coco_I.log  %log_path%\SUCCESS\human_pose_coco_I.log
        echo   human_pose_coco,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of human_pose_coco successfully!
)

rem mpii
rem train
python train.py --batch_size 8 --dataset mpii --num_epochs=1 > %log_path%/human_pose_mpii_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\human_pose_mpii_T.log  %log_path%\FAIL\human_pose_mpii_T.log
        echo   human_pose_mpii,train,FAIL  >> %log_path%\result.log
        echo   training of human_pose_mpii failed!
) else (
        move  %log_path%\human_pose_mpii_T.log  %log_path%\SUCCESS\human_pose_mpii_T.log
        echo   human_pose_mpii,train,SUCCESS  >> %log_path%\result.log
        echo   training of human_pose_mpii successfully!
 )
rem eval
python val.py --batch_size 8 --dataset mpii --checkpoint pretrained/pose-resnet50-mpii-384x384 --data_root data/mpii >%log_path%/human_pose_mpii_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\human_pose_mpii_E.log  %log_path%\FAIL\human_pose_mpii_E.log
        echo   human_pose_mpii,eval,FAIL  >> %log_path%\result.log
        echo   evaling of human_pose_mpii failed!
) else (
        move  %log_path%\human_pose_mpii_E.log  %log_path%\SUCCESS\human_pose_mpii_E.log
        echo   human_pose_mpii,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of human_pose_mpii successfully!
)
rem infer
python test.py --checkpoint pretrained/pose-resnet50-mpii-384x384 > %log_path%/human_pose_mpii_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\human_pose_mpii_I.log  %log_path%\FAIL\human_pose_mpii_I.log
        echo   human_pose_mpii,infer,FAIL  >> %log_path%\result.log
        echo   infering of human_pose_mpii failed!
) else (
        move  %log_path%\human_pose_mpii_I.log  %log_path%\SUCCESS\human_pose_mpii_I.log
        echo   human_pose_mpii,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of human_pose_mpii successfully!
) 