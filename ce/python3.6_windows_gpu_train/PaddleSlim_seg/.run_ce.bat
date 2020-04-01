@echo off
rem This file is only used for continuous evaluation.
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fraction_of_gpu_memory_to_use=0.98
set CUDA_VISIBLE_DEVICES=0
set sed="C:\Program Files\Git\usr\bin\sed.exe"
rem distill
%sed% -i s/"    SNAPSHOT_EPOCH: 5"/"    SNAPSHOT_EPOCH: 1"/g ./slim/distillation/cityscape.yaml
%sed% -i s/"    NUM_EPOCHS: 100"/"    NUM_EPOCHS: 1"/g ./slim/distillation/cityscape.yaml
%sed% -i s/"BATCH_SIZE: 16"/"BATCH_SIZE: 2"/g ./slim/distillation/cityscape.yaml
%sed% -i s/"    SNAPSHOT_EPOCH: 5"/"    SNAPSHOT_EPOCH: 1"/g ./slim/distillation/cityscape_teacher.yaml
%sed% -i s/"    NUM_EPOCHS: 100"/"    NUM_EPOCHS: 1"/g ./slim/distillation/cityscape_teacher.yaml
%sed% -i s/"BATCH_SIZE: 16"/"BATCH_SIZE: 2"/g ./slim/distillation/cityscape_teacher.yaml
python  ./slim/distillation/train_distill.py --log_steps 10 --cfg ./slim/distillation/cityscape.yaml --teacher_cfg ./slim/distillation/cityscape_teacher.yaml --use_gpu  --enable_ce > seg_distill.log 2>&1
type seg_distill.log|grep epoch|awk -F "[= ]" "END{print \"kpis\tdistill_train_loss\t\"$8}"|python _ce.py

rem prune
%sed% -i s/"    SNAPSHOT_EPOCH: 10"/"    SNAPSHOT_EPOCH: 1"/g configs/cityscape_fast_scnn.yaml
%sed% -i s/"  NUM_EPOCHS: 100"/"  NUM_EPOCHS: 1"/g configs/cityscape_fast_scnn.yaml
%sed% -i s/"BATCH_SIZE: 12"/"BATCH_SIZE: 4"/g configs/cityscape_fast_scnn.yaml
python -u ./slim/prune/train_prune.py --log_steps 10 --cfg configs/cityscape_fast_scnn.yaml --use_gpu SLIM.PRUNE_PARAMS 'learning_to_downsample/weights,learning_to_downsample/dsconv1/pointwise/weights,learning_to_downsample/dsconv2/pointwise/weights' SLIM.PRUNE_RATIOS [0.1,0.1,0.1] > seg_prune.log 2>&1
type seg_prune.log| grep epoch|awk -F "[ =]" "END{print \"kpis\tprune_train_loss\t\"$8}"|python _ce.py
python -u ./slim/prune/eval_prune.py --cfg configs/cityscape_fast_scnn.yaml --use_gpu TEST.TEST_MODEL snapshots/cityscape_fast_scnn/final > %log_path%/seg_prune_E.log 2>&1 
if not %errorlevel% == 0 (
        move  %log_path%\seg_prune_E.log  %log_path%\FAIL\seg_prune_E.log
        echo   seg_prune,eval,FAIL  >> %log_path%\result.log
        echo   eval of seg_prune failed!
) else (
        move  %log_path%\seg_prune_E.log  %log_path%\SUCCESS\seg_prune_E.log
        echo   seg_prune,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of seg_prune successfully!
 )
 
rem quant
set PYTHONPATH=pdseg;%PYTHONPATH%
python -u ./slim/quantization/train_quant.py --log_steps 10 --not_quant_pattern last_conv --cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml --use_gpu --enable_ce --do_eval TRAIN.PRETRAINED_MODEL_DIR "./pretrained_model/mobilenet_cityscapes/" TRAIN.MODEL_SAVE_DIR "./snapshots/mobilenetv2_quant" MODEL.DEEPLAB.ENCODER_WITH_ASPP False MODEL.DEEPLAB.ENABLE_DECODER False  TRAIN.SYNC_BATCH_NORM False SOLVER.LR 0.0001  TRAIN.SNAPSHOT_EPOCH 1 SOLVER.NUM_EPOCHS 1 BATCH_SIZE 2 >seg_quant.log 2>&1 
type  seg_quant.log|grep epoch|awk -F "[= ]" "END{print \"kpis\tquant_train_loss\t\"$8}"|python _ce.py
type  seg_quant.log|grep "EVAL"|grep step|awk -F "[= ]"  "END{print \"kpis\tquant_eval_loss\t\"$4}"|python _ce.py
python -u ./slim/quantization/eval_quant.py  --cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml  --use_gpu --not_quant_pattern last_conv  --convert TEST.TEST_MODEL "./snapshots/mobilenetv2_quant/best_model" MODEL.DEEPLAB.ENCODER_WITH_ASPP False MODEL.DEEPLAB.ENABLE_DECODER False TRAIN.SYNC_BATCH_NORM False BATCH_SIZE 8 > %log_path%/seg_quant_E.log
if not %errorlevel% == 0 (
        move  %log_path%\seg_quant_E.log  %log_path%\FAIL\seg_quant_E.log
        echo   seg_quant,eval,FAIL  >> %log_path%\result.log
        echo   eval of seg_quant failed!
) else (
        move  %log_path%\seg_quant_E.log  %log_path%\SUCCESS\seg_quant_E.log
        echo   seg_quant,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of seg_quant successfully!
 )
 
rem nas
%sed% -i s/"    SNAPSHOT_EPOCH: 10"/"    SNAPSHOT_EPOCH: 1"/g configs/deeplabv3p_mobilenetv2_cityscapes.yaml
%sed% -i s/"  NUM_EPOCHS: 100"/"  NUM_EPOCHS: 1"/g configs/deeplabv3p_mobilenetv2_cityscapes.yaml
python -u ./slim/nas/train_nas.py --log_steps 10 --cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml --enable_ce --use_gpu SLIM.NAS_PORT 23333 SLIM.NAS_ADDRESS ""  SLIM.NAS_SEARCH_STEPS 2 SLIM.NAS_START_EVAL_EPOCH -1 SLIM.NAS_IS_SERVER True SLIM.NAS_SPACE_NAME "MobileNetV2SpaceSeg" TRAIN.SNAPSHOT_EPOCH 1 SOLVER.NUM_EPOCHS 1 > seg_nas.log 2>&1
type seg_nas.log|grep epoch=|awk -F "[= ]" "END{print \"kpis\tnas_train_loss\t\"$8}" | python _ce.py
type seg_nas.log|grep "EVAL"|grep step|awk -F "[= ]"  "END{print \"kpis\tnas_eval_loss\t\"$4}" |python _ce.py


