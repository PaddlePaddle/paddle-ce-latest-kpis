@echo off
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fraction_of_gpu_memory_to_use=0.98
set CUDA_VISIBLE_DEVICES=0

set sed="C:\Program Files\Git\usr\bin\sed.exe"
rem distill
python slim/distillation/distill.py  -c configs/yolov3_mobilenet_v1_voc.yml -t configs/yolov3_r34_voc.yml --teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar -o batch_size=4 max_iters=200 snapshot_iter=200 save_dir=distill_output > detection_distill.log 2>&1
type detection_distill.log|grep loss|awk -F "[, ]" "END{print \"kpis\tdistill_train_loss\t\"$10}" | python _ce.py
python -u tools/eval.py -c configs/yolov3_mobilenet_v1_voc.yml  -o weights=distill_output/yolov3_mobilenet_v1_voc/model_final > %data_path%/detection_distill_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\detection_distill_E.log  %log_path%\FAIL\detection_distill_E.log
        echo   detection_distill,eval,FAIL  >> %log_path%\result.log
        echo   eval of detection_distill failed!
) else (
        move  %log_path%\detection_distill_E.log  %log_path%\SUCCESS\detection_distill_E.log
        echo   detection_distill,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of detection_distill successfully!
 )
python -u tools/infer.py -c configs/yolov3_mobilenet_v1_voc.yml --infer_img=demo/000000570688.jpg --output_dir=infer_output/  --draw_threshold=0.5 -o weights=distill_output/yolov3_mobilenet_v1_voc/model_final > %data_path%/detection_distill_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\detection_distill_I.log  %log_path%\FAIL\detection_distill_I.log
        echo   detection_distill,infer,FAIL  >> %log_path%\result.log
        echo   infer of detection_distill failed!
) else (
        move  %log_path%\detection_distill_I.log  %log_path%\SUCCESS\detection_distill_I.log
        echo   detection_distill,infer,SUCCISS  >> %log_path%\result.log
        echo   infer of detection_distill successfully!
)

rem prune
python slim/prune/prune.py -c configs/yolov3_mobilenet_v1_voc.yml --pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" --pruned_ratios="0.2,0.3,0.4" -o max_iters=200 snapshot_iter=200 save_dir=prune_output pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar  > detection_prune.log 2>&1 
type detection_prune.log|grep "iter: 180"|awk -F "[:,'']" "END{print \"kpis\tprune_train_loss\t\"$13}" | python _ce.py
python slim/prune/eval.py -c configs/yolov3_mobilenet_v1_voc.yml --pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" --pruned_ratios="0.2,0.3,0.4" -o weights=prune_output/yolov3_mobilenet_v1_voc/model_final > %data_path%/detection_prune_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\detection_prune_export.log  %log_path%\FAIL\detection_prune_export.log
        echo   detection_prune,eval,FAIL  >> %log_path%\result.log
        echo   eval of detection_prune failed!
) else (
        move  %log_path%\detection_prune_export.log  %log_path%\SUCCESS\detection_prune_export.log
        echo   detection_prune,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of detection_prune successfully!
 )
python slim/prune/export_model.py -c configs/yolov3_mobilenet_v1_voc.yml --output_dir prune_export_output --pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" --pruned_ratios="0.2,0.3,0.4" -o weights=prune_output/yolov3_mobilenet_v1_voc/model_final  > %data_path%/detection_prune_export.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\detection_prune_export.log  %log_path%\FAIL\detection_prune_export.log
        echo   detection_prune,export_model,FAIL  >> %log_path%\result.log
        echo   export_model of detection_prune failed!
) else (
        move  %log_path%\detection_prune_export.log  %log_path%\SUCCESS\detection_prune_export.log
        echo   detection_prune,export_model,SUCCISS  >> %log_path%\result.log
        echo   export_model of detection_prune successfully!
)
 
rem quant
python slim/quantization/train.py --not_quant_pattern yolo_output --eval -c ./configs/yolov3_mobilenet_v1.yml -o batch_size=2 save_dir=./quant_output/mobilenetv1 LearningRate.base_lr=0.0001 LearningRate.schedulers="[!PiecewiseDecay {gamma: 0.1, milestones: [10000]}]" pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar max_iters=200 snapshot_iter=200 > detection_quant.log 2>&1 
type detection_quant.log|grep "iter: 180"| awk -F "[:,'']" "{print \"kpis\tquant_train_loss\t\"$13"} | python _ce.py
python slim/quantization/eval.py --not_quant_pattern yolo_output  -c ./configs/yolov3_mobilenet_v1.yml -o weights=./quant_output/mobilenetv1/yolov3_mobilenet_v1/best_model > %data_path%/detection_quant_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\detection_quant_E.log  %log_path%\FAIL\detection_quant_E.log
        echo   detection_quant,eval,FAIL  >> %log_path%\result.log
        echo   eval of detection_quant failed!
) else (
        move  %log_path%\detection_quant_E.log  %log_path%\SUCCESS\detection_quant_E.log
        echo   detection_quant,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of detection_quant successfully!
 )
python slim/quantization/export_model.py --not_quant_pattern yolo_output  -c ./configs/yolov3_mobilenet_v1.yml --output_dir quant_export_output -o weights=./quant_output/mobilenetv1/yolov3_mobilenet_v1/best_model > %data_path%/detection_quant_export.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\detection_quant_export.log  %log_path%\FAIL\detection_quant_export.log
        echo   detection_quant,export_model,FAIL  >> %log_path%\result.log
        echo   export_model of detection_quant failed!
) else (
        move  %log_path%\detection_quant_export.log  %log_path%\SUCCESS\detection_quant_export.log
        echo   detection_quant,export_model,SUCCISS  >> %log_path%\result.log
        echo   export_model of detection_quant successfully!
 )
python slim/quantization/infer.py --not_quant_pattern yolo_output -c ./configs/yolov3_mobilenet_v1.yml --infer_dir ./demo -o weights=./quant_output/mobilenetv1/yolov3_mobilenet_v1/best_model  > %data_path%/detection_quant_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\detection_quant_I.log  %log_path%\FAIL\detection_quant_I.log
        echo   detection_quant,infer,FAIL  >> %log_path%\result.log
        echo   infer of detection_quant failed!
) else (
        move  %log_path%\detection_quant_I.log  %log_path%\SUCCESS\detection_quant_I.log
        echo   detection_quant,infer,SUCCISS  >> %log_path%\result.log
        echo   infer of detection_quant successfully!
 )
rem nas
rem %sed% -i s/"return [2, 1, 3, 8, 2, 1, 2, 1, 1]"/"return [2, 1, 3, 8, 2, 1, 2, 1, 0]"/g slim/nas/search_space/blazefacespace_nas.py
rem python -u slim/nas/train_nas.py -c slim/nas/blazeface.yml 






