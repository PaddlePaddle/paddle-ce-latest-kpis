#!/bin/bash

rm -rf *_factor.txt
export current_dir=$PWD
#  for lite models path
if [ ! -d "/ssd2/models_from_train" ];then
	mkdir /ssd2/models_from_train
fi
export models_from_train=/ssd2/models_from_train
print_info()
{
if [ $1 -ne 0 ];then
	echo -e "$2,train,FAIL"
else
	echo -e "$2,train,SUCCESS"
fi
}
export PYTHONPATH=`pwd`:$PYTHONPATH
# 1 distillation
# 1card max_iters=8000  2h
# 8card max_iters=2000  30min
model=dist_yolov3_v1_voc_ce_1card
if [ -d "output" ];then
    rm -rf output
fi
dete_dist_yolov3_v1()
{
    python ./slim/distillation/distill.py \
    -c configs/yolov3_mobilenet_v1_voc.yml \
    -t configs/yolov3_r34_voc.yml \
    --save_inference true \
    -o max_iters=$1 --teacher_pretrained https://paddlemodels.bj.bcebos.com/object_detection/yolov3_r34_voc.tar
}
CUDA_VISIBLE_DEVICES=0 dete_dist_yolov3_v1 8000 >dete_dist_yolov3_v1_1card 2>&1
cat dete_dist_yolov3_v1_1card|grep Best | awk -F ' ' 'END{print "kpis\tdist_yolov3_v1_bestap_1card\t"$7}'|tr -d ',' | python _ce.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dete_dist_yolov3_v1 2000 >dete_dist_yolov3_v1_8card 2>&1
cat dete_dist_yolov3_v1_8card|grep Best | awk -F ' ' 'END{print "kpis\tdist_yolov3_v1_bestap_8card\t"$7}'|tr -d ',' | python _ce.py

# 1.2 infer
model=dist_yolov3_mobilenet_v1_voc_infer
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u tools/infer.py \
-c configs/yolov3_mobilenet_v1_voc.yml \
--infer_img=demo/000000570688.jpg \
--output_dir=infer_output/ \
--draw_threshold=0.5 >${model} 2>&1
print_info $? ${model}
#1.3 eval
model=dist_yolov3_mobilenet_v1_voc_eval
python -u tools/eval.py -c configs/yolov3_mobilenet_v1_voc.yml >${model} 2>&1
print_info $? ${model}

if [ -d "output" ];then
	mv  output dist_output
fi
# 将save的模型转存供lite使用
mkdir dete_dist_yolov3_v1
cp dist_output/yolov3_mobilenet_v1_voc/infer/* dete_dist_yolov3_v1/
cp -r dete_dist_yolov3_v1 ${models_from_train}/

# 2.1 quan train  35min
dete_quan_yolov3_v1()
{
    python slim/quantization/train.py --not_quant_pattern yolo_output \
    --eval \
    -c ./configs/yolov3_mobilenet_v1.yml \
    -o max_iters=2000  snapshot_iter=1000 \
    LearningRate.base_lr=0.000001 \
    LearningRate.schedulers='[!PiecewiseDecay {gamma: 0.1, milestones: [40000]}]' \
    pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1.tar
}
CUDA_VISIBLE_DEVICES=0 dete_quan_yolov3_v1 > dete_quan_yolov3_v1_1card 2>&1
cat dete_quan_yolov3_v1_1card|grep Best | awk -F ' ' 'END{print "kpis\tquan_yolov3_v1_bestap_1card\t"$7}'|tr -d ',' | python _ce.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dete_quan_yolov3_v1 > dete_quan_yolov3_v1_8card 2>&1
cat dete_quan_yolov3_v1_8card|grep Best | awk -F ' ' 'END{print "kpis\tquan_yolov3_v1_bestap_8card\t"$7}'|tr -d ',' | python _ce.py

cp ./configs/dcn/yolov3_r50vd_dcn_obj365_pretrained_coco.yml ./configs/
cp ./configs/dcn/yolov3_enhance_reader.yml ./configs/
# 需要修改路径保证下面eval正常循环
sed -i "s/yolov3_r50vd_dcn_db_obj365_pretrained_coco/yolov3_r50vd_dcn_obj365_pretrained_coco/g" ./configs/yolov3_r50vd_dcn_obj365_pretrained_coco.yml
dete_quan_yolov3()
{
    python slim/quantization/train.py --not_quant_pattern yolo_output \
    --eval \
    -c ./configs/$1.yml \
    -o max_iters=2000  snapshot_iter=1000 \
    LearningRate.base_lr=0.0001 \
    LearningRate.schedulers='[!PiecewiseDecay {gamma: 0.1, milestones: [40000]}]' \
    pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/$1.tar
}
quan_models=(yolov3_r34 yolov3_r50vd_dcn_obj365_pretrained_coco)
for i in $(seq 0 1); do
    CUDA_VISIBLE_DEVICES=0 dete_quan_yolov3 ${quan_models[$i]} > dete_quan_${quan_models[$i]}_1card 2>&1
    cat dete_quan_${quan_models[$i]}_1card|grep Best | awk -F ' ' 'END{print "kpis\tquan_""'${quan_models[$i]}_bestap_1card'""\t"$7}'|tr -d ',' | python _ce.py
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dete_quan_yolov3 ${quan_models[$i]} > dete_quan_${quan_models[$i]}_8card 2>&1
    cat dete_quan_${quan_models[$i]}_8card|grep Best | awk -F ' ' 'END{print "kpis\tquan_""'${quan_models[$i]}_bestap_8card'""\t"$7}'|tr -d ',' | python _ce.py
done
# 2.2 eval
quan_models=(yolov3_mobilenet_v1 yolov3_r34 yolov3_r50vd_dcn_obj365_pretrained_coco)
model=dete_quan_eval
for i in $(seq 0 2); do
CUDA_VISIBLE_DEVICES=0 python slim/quantization/eval.py \
    --not_quant_pattern yolo_output  \
    -c ./configs/${quan_models[$i]}.yml >${model}_${quan_models[$i]} 2>&1
print_info $? ${model}_${quan_models[$i]}
done
# 2.3 infer
model=dete_quan_infer
for i in $(seq 0 2); do
    CUDA_VISIBLE_DEVICES=0  python slim/quantization/infer.py \
    --not_quant_pattern yolo_output \
    -c ./configs/${quan_models[$i]}.yml \
    --infer_dir ./demo  >${model}_${quan_models[$i]} 2>&1
print_info $? ${model}_${quan_models[$i]}
done

# 2.4 export
model=dete_quan_export
mkdir dete_quan_export_float
for i in $(seq 0 2); do
    CUDA_VISIBLE_DEVICES=0 python slim/quantization/export_model.py \
    --not_quant_pattern yolo_output  \
    -c ./configs/${quan_models[$i]}.yml \
    --output_dir ./quan_export/dete_${quan_models[$i]} >${model}_${quan_models[$i]} 2>&1
print_info $? ${model}_${quan_models[$i]}
mkdir dete_quan_export_float/dete_${quan_models[$i]}
cp ./quan_export/dete_${quan_models[$i]}/float/* ./dete_quan_export_float/dete_${quan_models[$i]}/
done
# export_model后的结果转存
cp -r dete_quan_export_float/* ${models_from_train}/
# yolov3_r50vd lite 暂不支持
rm -rf ${models_from_train}/dete_yolov3_r50vd_dcn_obj365_pretrained_coco

if [ -d "output" ];then
	mv  output quan_output
fi

# 3 prune
# 3.1 prune train
cp ./configs/dcn/yolov3_r50vd_dcn.yml ./configs/
sed -i $"s?_READER_: '../yolov3_reader.yml'?_READER_: './yolov3_reader.yml'?g" ./configs/yolov3_r50vd_dcn.yml
prune_models=(yolov3_mobilenet_v1 yolov3_mobilenet_v1_voc yolov3_r50vd_dcn)
dete_prune_yolov3_iter2000()
{
    python slim/prune/prune.py \
    -c configs/$1.yml \
    --pruned_params="yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights,yolo_block.0.1.1.conv.weights,yolo_block.0.2.conv.weights,yolo_block.0.tip.conv.weights,yolo_block.1.0.0.conv.weights,yolo_block.1.0.1.conv.weights,yolo_block.1.1.0.conv.weights,yolo_block.1.1.1.conv.weights,yolo_block.1.2.conv.weights,yolo_block.1.tip.conv.weights,yolo_block.2.0.0.conv.weights,yolo_block.2.0.1.conv.weights,yolo_block.2.1.0.conv.weights,yolo_block.2.1.1.conv.weights,yolo_block.2.2.conv.weights,yolo_block.2.tip.conv.weights" \
    --pruned_ratios="0.1,0.2,0.2,0.2,0.2,0.1,0.2,0.3,0.3,0.3,0.2,0.1,0.3,0.4,0.4,0.4,0.4,0.3" \
    --eval \
    -o max_iters=2000 pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/$1.tar
}
dete_prune_yolov3_iter8000()
{
    python slim/prune/prune.py \
    -c configs/$1.yml \
    --pruned_params="yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights,yolo_block.0.1.1.conv.weights,yolo_block.0.2.conv.weights,yolo_block.0.tip.conv.weights,yolo_block.1.0.0.conv.weights,yolo_block.1.0.1.conv.weights,yolo_block.1.1.0.conv.weights,yolo_block.1.1.1.conv.weights,yolo_block.1.2.conv.weights,yolo_block.1.tip.conv.weights,yolo_block.2.0.0.conv.weights,yolo_block.2.0.1.conv.weights,yolo_block.2.1.0.conv.weights,yolo_block.2.1.1.conv.weights,yolo_block.2.2.conv.weights,yolo_block.2.tip.conv.weights" \
    --pruned_ratios="0.1,0.2,0.2,0.2,0.2,0.1,0.2,0.3,0.3,0.3,0.2,0.1,0.3,0.4,0.4,0.4,0.4,0.3" \
    --eval \
    -o max_iters=8000 pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/$1.tar
}
for i in $(seq 0 2); do
    CUDA_VISIBLE_DEVICES=0 dete_prune_yolov3_iter8000  ${prune_models[$i]} > dete_prune_${prune_models[$i]}_1card 2>&1
    cat dete_prune_${prune_models[$i]}_1card|grep Best | awk -F ' ' 'END{print "kpis\tprune_""'${prune_models[$i]}_bestap_1card'""\t"$7}'|tr -d ',' | python _ce.py
    sleep 20
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 dete_prune_yolov3_iter2000 ${prune_models[$i]} > dete_prune_${prune_models[$i]}_8card 2>&1
    cat dete_prune_${prune_models[$i]}_8card|grep Best | awk -F ' ' 'END{print "kpis\tprune_""'${prune_models[$i]}_bestap_8card'""\t"$7}'|tr -d ',' | python _ce.py
done

# 3.2 prune eval
model=dete_prune_eval
for i in $(seq 0 2); do
    python slim/prune/eval.py \
    -c configs/${prune_models[$i]}.yml \
    --pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" \
    --pruned_ratios="0.2,0.3,0.4" >${model}_${prune_models[$i]} 2>&1
print_info $? ${model}_${prune_models[$i]} 2>&1
done
#3.3 prune export
model=dete_prune_export
for i in $(seq 0 2); do
    python slim/prune/export_model.py \
    -c configs/${prune_models[$i]}.yml \
    --pruned_params "yolo_block.0.0.0.conv.weights,yolo_block.0.0.1.conv.weights,yolo_block.0.1.0.conv.weights" \
    --pruned_ratios="0.2,0.3,0.4" >${model}_${prune_models[$i]} 2>&1
print_info $? ${model}_${prune_models[$i]} 2>&1
# 转存，这里export的结果还在原output中
mkdir slim_dete_${prune_models[$i]}
cp output/${prune_models[$i]}/__model__ ./slim_dete_${prune_models[$i]}/
cp output/${prune_models[$i]}/__params__ ./slim_dete_${prune_models[$i]}/
cp -r slim_dete_${prune_models[$i]} ${models_from_train}/
done

# 3.4 prune infer
model=dete_prune_infer
for i in $(seq 0 2); do
    CUDA_VISIBLE_DEVICES=0  python -u tools/infer.py \
    -c configs/${prune_models[$i]}.yml \
    --infer_img=demo/000000570688.jpg \
    --output_dir=infer_output/ \
    --draw_threshold=0.5 >${model}_${prune_models[$i]} 2>&1
print_info $? ${model}_${prune_models[$i]}
done

if [ -d "output" ];then
	mv  output prune_output
fi

