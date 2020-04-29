#!/bin/bash

export PYTHONPATH=`pwd`:$PYTHONPATH
#train/eval/infer/export
train_model(){
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python -u tools/train.py -c ${config_dir}/${model}.yml --enable_ce=True -o max_iters=${max_iters} TrainReader.shuffle=false --eval >${model}_log 2>&1
    sed -i "s/'loss'/'${model}_loss'/g" ${model}_log
    sed -i "s/time/${model}_time /g" ${model}_log
    cat ${model}_log | grep "loss" | tail -1 | tr "," " " | tr "'" " "| python _ce.py
}
eval_model(){
    export CUDA_VISIBLE_DEVICES=0
    python tools/${eval_method}.py -c ${config_dir}/${model}.yml -o weights=https://paddlemodels.bj.bcebos.com/object_detection/${model}.tar >${model}_eval_log 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model},eval,FAIL"
    else
        echo -e "${model},eval,SUCCESS"
    fi
}
infer_model(){
    python tools/infer.py -c ${config_dir}/${model}.yml --infer_img=demo/000000570688.jpg --output_dir=${model} -o weights=https://paddlemodels.bj.bcebos.com/object_detection/${model}.tar >${model}_infer_log 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model},infer,FAIL"
    else
        echo -e "${model},infer,SUCCESS"
    fi
}
export_model(){
    python tools/export_model.py -c ${config_dir}/${model}.yml --output_dir=./inference_model -o weights=https://paddlemodels.bj.bcebos.com/object_detection/${model}.tar TestReader.inputs_def.image_shape=[3,640,640] >${model}_export_log 2>&1
    if [ $? -ne 0 ];then
        echo -e "${model},export,FAIL"
    else
        echo -e "${model},export,SUCCESS"
    fi
}
model_list='cascade_rcnn_r50_fpn_1x faster_rcnn_r50_fpn_1x mask_rcnn_r50_fpn_1x mask_rcnn_r101_vd_fpn_1x retinanet_r50_fpn_1x yolov3_r50vd_dcn_obj365_pretrained_coco yolov3_darknet yolov3_r34_voc blazeface_nas'
for model in ${model_list}
do
if [[ ${model} == yolov3_r50vd_dcn_obj365_pretrained_coco || ${model} == yolov3_darknet || ${model} == yolov3_r34_voc || ${model} == blazeface_nas ]];then
    max_iters=200
else
    max_iters=1000
fi
if [[ ${model} == yolov3_r50vd_dcn_obj365_pretrained_coco ]];then
    config_dir=configs/dcn
    eval_method=eval
elif [[ ${model} == blazeface_nas ]];then
    config_dir=configs/face_detection
    eval_method=face_eval
else
    config_dir=configs
    eval_method=eval
fi
train_model
eval_model
infer_model
export_model
done

