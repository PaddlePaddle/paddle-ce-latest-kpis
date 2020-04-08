#!/bin/bash

# This file is only used for continuous evaluation.

export ce_mode=1
rm -rf *_factor.txt
export current_dir=$PWD

#set result dir
cd ${current_dir}
if [ ! -d "result" ];then
	mkdir result
fi
result_path=${current_dir}"/result"
cd ${result_path}
if [ -d "result.log" ];then
	rm -rf result.log
fi
#set log dir
cd ${current_dir}
if [ -d "logs" ];then
    rm -rf logs
fi
mkdir logs && cd logs
mkdir SUCCESS
mkdir FAIL
log_path=${current_dir}"/logs"

# 1 uniform_prune
cd ${current_dir}
export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=7
if [ -d 'checkpoints' ]; then
    rm -rf checkpoints
fi
sed -i "s/epoch: 200/epoch: 1/g" configs/filter_pruning_uniform.yaml
python compress.py \
    --model "MobileNet" \
    --pretrained_model ./pretrain/MobileNetV1_pretrained \
    --compress_config ./configs/filter_pruning_uniform.yaml >$PWD/1_filter_pruning_uniform_run.log 2>&1
echo "uniform_prune log ok"
cd ${current_dir}
cat 1_filter_pruning_uniform_run.log|grep Final|awk -F '[' 'END{print$3}' | tr -d "[|]|,"|awk -F ' ' 'END{print "kpis\tslim_uniform_prune_test_acc_top1\t"$1}END{print "kpis\tslim_uniform_prune_test_acc_top5\t"$2}'| python _ce.py

# 2 sen_prune_v1
export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=7
if [ -d 'checkpoints' ]; then
    rm -rf checkpoints
fi
sed -i "s/epoch: 200/epoch: 1/g" configs/filter_pruning_sen.yaml
python -u compress.py \
        --model MobileNet \
        --pretrained_model ./pretrain/MobileNetV1_pretrained \
        --compress_config ./configs/filter_pruning_sen.yaml >$PWD/2_filter_pruning_sen_run.log 2>&1
echo "sen_prune log ok"
cd ${current_dir}
cat 2_filter_pruning_sen_run.log|grep Final|awk -F '[' 'END{print$3}' | tr -d "[|]|,"|awk -F ' ' 'END{print "kpis\tslim_sen_prune_test_acc_top1\t"$1}END{print "kpis\tslim_sen_prune_test_acc_top5\t"$2}'| python _ce.py

# 3 quan_v1
export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=7
if [ -d 'checkpoints_quan' ]; then
    rm -rf checkpoints_quan
fi
sed -i "s/end_epoch: 19/end_epoch: 1/g" configs/quantization.yaml
sed -i "s/epoch: 20/epoch: 1/g" configs/quantization.yaml

python compress.py \
        --batch_size 64 \
        --model "MobileNet" \
        --pretrained_model ./pretrain/MobileNetV1_pretrained \
        --quant_only True \
        --compress_config ./configs/quantization.yaml >$PWD/3_quan_run.log 2>&1
echo "quan_v1 log ok"
cd ${current_dir}
cat 3_quan_run.log|grep Final|awk -F '[' 'END{print$3}' | tr -d "[|]|,"|awk -F ' ' 'END{print "kpis\tslim_quan_v1_test_acc_top1\t"$1}END{print "kpis\tslim_quan_v1_test_acc_top5\t"$2}'| python _ce.py

# 4 dist_v1
export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd ${current_dir}
sed -i "s/end_epoch: 130/end_epoch: 1/g" configs/mobilenetv1_resnet50_distillation.yaml
sed -i "s/epoch: 130/epoch: 1/g" configs/mobilenetv1_resnet50_distillation.yaml

if [ -d 'checkpoints' ]; then
    rm -rf checkpoints
fi
python compress.py \
        --enable_ce True \
        --model "MobileNet" \
        --teacher_model "ResNet50" \
        --teacher_pretrained_model ./pretrain/ResNet50_pretrained \
        --compress_config ./configs/mobilenetv1_resnet50_distillation.yaml  >$PWD/4_dist_v1_run.log 2>&1
echo "dist_v1 log ok"
cd ${current_dir}
cat 4_dist_v1_run.log|grep Final|awk -F '[' 'END{print$3}' | tr -d "[|]|,"|awk -F ' ' 'END{print "kpis\tslim_dist_v1_test_acc_top1\t"$1}END{print "kpis\tslim_dist_v1_test_acc_top5\t"$2}'| python _ce.py

# 5 class_quan_v2
export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd ${current_dir}/classification/quantization

if [ -d 'checkpoints' ]; then
    rm -rf checkpoints
fi
sed -i "s/end_epoch: 29/end_epoch: 0/g" configs/mobilenet_v2.yaml
sed -i "s/epoch: 30/epoch: 1/g" configs/mobilenet_v2.yaml
python compress.py \
    --model "MobileNetV2" \
    --use_gpu 1 \
    --batch_size 256 \
    --pretrained_model ../pretrain/MobileNetV2_pretrained \
    --config_file  "./configs/mobilenet_v2.yaml" >${current_dir}/5_class_quan_v2_run.log 2>&1
cd ${current_dir}
cat 5_class_quan_v2_run.log|grep Final|awk -F '[' 'END{print$3}' | tr -d "[|]|,"|awk -F ' ' 'END{print "kpis\tclass_quan_v2_test_acc_top1\t"$1}END{print "kpis\tclass_quan_v2_test_acc_top5\t"$2}'| python _ce.py
# # classification_quan_freeze and infer
cd ${current_dir}/classification/quantization
quan_model=mobilenet_v2
time (python freeze.py \
        --model_path ./checkpoints/${quan_model}/0/eval_model \
        --weight_quant_type  abs_max\
        --save_path ./freeze/${quan_model} >${log_path}/class_quan_freeze_${quan_model}.log) >>${log_path}/class_quan_freeze_${quan_model}.log 2>&1
    if [ $? -ne 0 ];then
	    mv ${log_path}/class_quan_freeze_${quan_model}.log ${log_path}/FAIL/class_quan_freeze_${quan_model}.log
	    echo -e "class_quan_freeze_${quan_model},freeze,FAIL" >>${result_path}/result.log;
    else
	    mv ${log_path}/class_quan_freeze_${quan_model}.log ${log_path}/SUCCESS/class_quan_freeze_${quan_model}.log
	    echo -e "class_quan_freeze_${quan_model},freeze,SUCCESS" >>${result_path}/result.log
    fi
cd ${current_dir}/classification/quantization/output/mobilenet_v2/float
mv model __model__.infer
mv weights __params__
cd ${current_dir}/classification/quantization/freeze/mobilenet_v2/float
mv model __model__.infer
mv weights __params__
# 6 class_prune_v1
export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd ${current_dir}/classification/pruning
if [ -d 'checkpoints' ]; then
    rm -rf checkpoints
fi
sed -i "s/epoch: 121/epoch: 1/g" configs/mobilenet_v1.yaml
python -u compress.py \
    --enable_ce True \
    --model "MobileNet" \
    --use_gpu 1 \
    --batch_size 256 \
    --total_images 1281167 \
    --lr_strategy "piecewise_decay" \
    --num_epochs 1 \
    --lr 0.1 \
    --l2_decay 3e-5 \
    --pretrained_model ../pretrain/MobileNetV1_pretrained \
    --config_file ./configs/mobilenet_v1.yaml >${current_dir}/6_class_prune_v1_run.log 2>&1

cd ${current_dir}
cat 6_class_prune_v1_run.log|grep Final|awk -F '[' 'END{print$3}' | tr -d "[|]|,"|awk -F ' ' 'END{print "kpis\tclass_prune_v1_test_acc_top1\t"$1}END{print "kpis\tclass_prune_v1_test_acc_top5\t"$2}'| python _ce.py
# 7 dist_resnet34
export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=4,5,6,7
cd ${current_dir}/classification/distillation
sed -i "s/end_epoch: 130/end_epoch: 1/g" configs/resnet34_resnet50_distillation.yaml
sed -i "s/epoch: 130/epoch: 1/g" configs/resnet34_resnet50_distillation.yaml
if [ -d 'checkpoints' ]; then
    rm -rf checkpoints
fi
python -u compress.py \
    --enable_ce True \
    --model "ResNet34" \
    --teacher_model "ResNet50" \
    --teacher_pretrained_model ../pretrain/ResNet50_pretrained \
    --compress_config ./configs/resnet34_resnet50_distillation.yaml >${current_dir}/7_class_resnet34_run.log 2>&1

cd ${current_dir}
cat 7_class_resnet34_run.log|grep Final|awk -F '[' 'END{print$3}' | tr -d "[|]|,"|awk -F ' ' 'END{print "kpis\tclass_dist_resnet34_test_acc_top1\t"$1}END{print "kpis\tclass_dist_resnet34_test_acc_top5\t"$2}'| python _ce.py

# infer eval
model_list=(quan_mobilenet_v2  quan_mobilenet_v2_freeze prune_mobilenet_v1 dist_resnet34_resnet50)
model_paths=(./quantization/output/mobilenet_v2/float ./quantization/freeze/mobilenet_v2/float ./pruning/checkpoints/mobilenet_v1/0/eval_model ./distillation/checkpoints/0/eval_model)
cd ${current_dir}/classification
for i in $(seq 0 3); do
    echo $i ${model_paths[$i]} ${model_list[$i]}
    time (python infer.py \
        --use_gpu 0 \
        --model_path ${model_paths[$i]} \
        --model_name __model__.infer \
        --params_name __params__ >${log_path}/${model_list[$i]}_infer.log) >>${log_path}/${model_list[$i]}_infer.log 2>&1
    if [ $? -ne 0 ];then
	    mv ${log_path}/${model_list[$i]}_infer.log ${log_path}/FAIL/${model_list[$i]}_infer.log
	    echo -e "${model_list[$i]}_infer,infer,FAIL" >>${result_path}/result.log;
    else
	    mv ${log_path}/${model_list[$i]}}_infer.log ${log_path}/SUCCESS/${model_list[$i]}_infer.log
	    echo -e "${model_list[$i]}_infer,infer,SUCCESS" >>${result_path}/result.log
    fi
    time (python eval.py \
        --use_gpu True \
        --model_path ${model_paths[$i]} \
        --model_name __model__.infer \
        --params_name __params__ >${log_path}/${model_list[$i]}_eval.log) >>${log_path}/${model_list[$i]}_eval.log 2>&1
    if [ $? -ne 0 ];then
	    mv ${log_path}/${model_list[$i]}_eval.log ${log_path}/FAIL/${model_list[$i]}_eval.log
	    echo -e "${model_list[$i]}_eval,eval,FAIL" >>${result_path}/result.log;
    else
	    mv ${log_path}/${model_list[$i]}_eval.log ${log_path}/SUCCESS/${model_list[$i]}_eval.log
	    echo -e "${model_list[$i]}_eval,eval,SUCCESS" >>${result_path}/result.log
    fi
done
