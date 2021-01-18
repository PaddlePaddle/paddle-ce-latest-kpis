#!/bin/bash

rm -rf *_factor.txt
cd legacy
export current_dir=$PWD

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/FAIL_$2.log
    echo -e "\033[31m ${log_path}/$2_FAIL \033[0m"
else
    mv ${log_path}/$2 ${log_path}/SUCCESS_$2.log
    echo -e "\033[32m ${log_path}/$2_SUCCESS \033[0m"
fi
}
#copy_for_lite ${model_name} ${models_from_train}
copy_for_lite(){
if [ -d $2/$1 ]; then
    rm -rf $2/$1
fi
if [ "$(ls -A $1)" ];then
   tar -czf $1.tar.gz $1
   cp $1.tar.gz $2/
   echo "\033[32m -----$1 copy for lite SUCCESS----- \033[0m"
else
   echo "\033[31m -----$1 is empty----- \033[0m"
fi
}
cudaid1=${card1:=2} # use 0-th card as default
cudaid8=${card8:=0,1,2,3,4,5,6,7} # use 0-th card as default
cudaid2=${card2:=2,3} # use 0-th card as default
####################################################
export PYTHONPATH=$PYTHONPATH:./pdseg
# 1 distillation
# 1card
# 8card
sed -i "s/    NUM_EPOCHS: 100/    NUM_EPOCHS: 1/g" ./slim/distillation/cityscape_teacher.yaml
sed -i "s/    NUM_EPOCHS: 100/    NUM_EPOCHS: 1/g" ./slim/distillation/cityscape.yaml
sed -i "s/    SNAPSHOT_EPOCH: 5/    SNAPSHOT_EPOCH: 1/g" ./slim/distillation/cityscape.yaml
# 1.1 dist train
seg_dist_Dv3_xception_mobilenet()
{
python -m paddle.distributed.launch ./slim/distillation/train_distill.py \
--log_steps 10 \
--cfg ./slim/distillation/cityscape.yaml \
--teacher_cfg ./slim/distillation/cityscape_teacher.yaml \
--use_gpu \
--do_eval \
BATCH_SIZE 7
}
CUDA_VISIBLE_DEVICES=${cudaid1} seg_dist_Dv3_xception_mobilenet 1>${log_path}/seg_dist_Dv3_xception_mobilenet_1card 2>&1
cat ${log_path}/seg_dist_Dv3_xception_mobilenet_1card |grep image=50 |awk -F ' |=' 'END{print "kpis\tseg_dist_Dv3_xception_mobilenet_acc_1card\t"$4"\tkpis\tseg_dist_Dv3_xception_mobilenet_IoU_1card\t"$6}' | python _ce.py
CUDA_VISIBLE_DEVICES=${cudaid8} seg_dist_Dv3_xception_mobilenet 1>${log_path}/seg_dist_Dv3_xception_mobilenet_8card 2>&1
cat ${log_path}/seg_dist_Dv3_xception_mobilenet_8card |grep image=50 |awk -F ' |=' 'END{print "kpis\tseg_dist_Dv3_xception_mobilenet_acc_8card\t"$4"\tkpis\tseg_dist_Dv3_xception_mobilenet_IoU_8card\t"$6}' | python _ce.py

# 1.2  dist export
model=seg_dist_Dv3_xception_mobilenet_export
CUDA_VISIBLE_DEVICES=${cudaid1} python pdseg/export_model.py --cfg ./slim/distillation/cityscape.yaml \
TEST.TEST_MODEL ./snapshots/cityscape_mbv2_kd_e100_1/final >${log_path}/${model} 2>&1
print_info $? ${model}
#infer
# 1.3  dist infer environment
cd ./deploy/python
pip install -r requirements.txt
yum install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
model=seg_dist_Dv3_xception_mobilenet_infer
cd ${current_dir}

python ./deploy/python/infer.py --conf=./freeze_model/deploy.yaml \
--input_dir=./test_img >${log_path}/${model} 2>&1
print_info $? ${model}
mv freeze_model seg_dist_v3_xception_mobilenet
# lite op not support
#cp -r seg_dist_v3_xception_mobilenet  ${models_from_train}/
# 1.4 dist eval
model=seg_dist_Dv3_xception_mobilenet_eval
CUDA_VISIBLE_DEVICES=${cudaid1} python pdseg/eval.py --use_gpu --cfg ./slim/distillation/cityscape.yaml \
TEST.TEST_MODEL ./snapshots/cityscape_mbv2_kd_e100_1/final >${log_path}/${model} 2>&1
print_info $? ${model}
if [ -d "snapshots" ];then
	mv  snapshots seg_dist_snapshots
fi


# 2 seg quan
seg_quan_Deeplabv3_v2 ()
{
python -u ./slim/quantization/train_quant.py \
--log_steps 10 \
--not_quant_pattern last_conv \
--cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml \
--use_gpu \
--do_eval \
TRAIN.PRETRAINED_MODEL_DIR "./pretrained_model/mobilenet_cityscapes/" \
TRAIN.MODEL_SAVE_DIR "./snapshots/mobilenetv2_quant" \
MODEL.DEEPLAB.ENCODER_WITH_ASPP False \
MODEL.DEEPLAB.ENABLE_DECODER False \
TRAIN.SYNC_BATCH_NORM False \
SOLVER.LR 0.0001 \
TRAIN.SNAPSHOT_EPOCH 1 \
SOLVER.NUM_EPOCHS 1 \
BATCH_SIZE 7
}
CUDA_VISIBLE_DEVICES=${cudaid1} seg_quan_Deeplabv3_v2 1>${log_path}/seg_quan_Deeplabv3_v2_1card 2>&1
cat ${log_path}/seg_quan_Deeplabv3_v2_1card |grep image=50 |awk -F ' |=' 'END{print "kpis\tseg_quan_Dv3_v2_acc_1card\t"$4"\tkpis\tseg_quan_Dv3_v2_IoU_1card\t"$6}' | python _ce.py
CUDA_VISIBLE_DEVICES=${cudaid8} seg_quan_Deeplabv3_v2 1>${log_path}/seg_quan_Deeplabv3_v2_8card 2>&1
cat ${log_path}/seg_quan_Deeplabv3_v2_8card |grep image=50 |awk -F ' |=' 'END{print "kpis\tseg_quan_Dv3_v2_acc_8card\t"$4"\tkpis\tseg_quan_Dv3_v2_IoU_8card\t"$6}' | python _ce.py

# 2.2  seg quan export
model=seg_quan_Dv3_v2_export
CUDA_VISIBLE_DEVICES=${cudaid1} python -u ./slim/quantization/export_model.py \
--not_quant_pattern last_conv  \
--cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml  \
TEST.TEST_MODEL "./snapshots/mobilenetv2_quant/best_model" \
MODEL.DEEPLAB.ENCODER_WITH_ASPP False \
MODEL.DEEPLAB.ENABLE_DECODER False \
TRAIN.SYNC_BATCH_NORM False \
SLIM.PREPROCESS True >${log_path}/${model} 2>&1
print_info $? ${model}

# 2.3 quan infer
# infer environment
#model=seg_quan_Dv3_v2_infer
#CUDA_VISIBLE_DEVICES=${cudaid1} python ./deploy/python/infer.py --conf=./freeze_model/deploy.yaml \
#--input_dir=./test_img >${log_path}/${model} 2>&1
#print_info $? ${model}
# for lite
mv freeze_model seg_quan_Dv3_v2_combined
copy_for_lite seg_quan_Dv3_v2_combined ${models_from_train}

# 2.4 quan_eval
model=seg_quan_Dv3_v2_eval_quant
CUDA_VISIBLE_DEVICES=${cudaid1} python -u ./slim/quantization/eval_quant.py \
--cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml \
--use_gpu --not_quant_pattern last_conv --convert \
TEST.TEST_MODEL "./snapshots/mobilenetv2_quant/best_model" \
MODEL.DEEPLAB.ENCODER_WITH_ASPP False \
MODEL.DEEPLAB.ENABLE_DECODER False \
TRAIN.SYNC_BATCH_NORM False \
BATCH_SIZE 7 >${log_path}/${model} 2>&1
print_info $? ${model}

if [ -d "snapshots" ];then
	mv  snapshots seg_quant_snapshots
fi

# 3 prune
# 3.1 prune train
seg_prune_Fast_SCNN (){
python -u ./slim/prune/train_prune.py \
--log_steps 10 \
--cfg configs/cityscape_fast_scnn.yaml \
--use_gpu \
--do_eval \
BATCH_SIZE 7 \
SLIM.PRUNE_PARAMS 'learning_to_downsample/weights,learning_to_downsample/dsconv1/pointwise/weights,learning_to_downsample/dsconv2/pointwise/weights' \
SLIM.PRUNE_RATIOS '[0.1,0.1,0.1]' \
TRAIN.SNAPSHOT_EPOCH 1 \
SOLVER.NUM_EPOCHS 1
}
CUDA_VISIBLE_DEVICES=${cudaid1} seg_prune_Fast_SCNN 1>${log_path}/seg_prune_Fast_SCNN_1card 2>&1
cat ${log_path}/seg_prune_Fast_SCNN_1card |grep image=50 |awk -F ' |=' 'END{print "kpis\tseg_prune_Fast_SCNN_acc_1card\t"$4"\tkpis\tseg_prune_Fast_SCNN_IoU_1card\t"$6}' | python _ce.py
CUDA_VISIBLE_DEVICES=${cudaid8} seg_prune_Fast_SCNN 1>${log_path}/seg_prune_Fast_SCNN_8card 2>&1
cat ${log_path}/seg_prune_Fast_SCNN_8card |grep image=50 |awk -F ' |=' 'END{print "kpis\tseg_prune_Fast_SCNN_acc_8card\t"$4"\tkpis\tseg_prune_Fast_SCNN_IoU_8card\t"$6}' | python _ce.py

# 3.2 seg/prune eval
model=seg_prune_Fast_SCNN_eval
CUDA_VISIBLE_DEVICES=${cudaid1} python -u ./slim/prune/eval_prune.py \
--cfg configs/cityscape_fast_scnn.yaml \
--use_gpu \
TEST.TEST_MODEL ./snapshots/cityscape_fast_scnn/final >${log_path}/${model} 2>&1
print_info $? ${model}

if [ -d "snapshots" ];then
	mv  snapshots seg_prune_snapshots
fi

# 3.3 prune no infer

# 4 nas
nas_train(){
python -u ./slim/nas/train_nas.py \
--log_steps 10 \
--cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml \
--use_gpu \
SLIM.NAS_PORT $1 \
SLIM.NAS_ADDRESS "" \
SLIM.NAS_SEARCH_STEPS 2 \
SLIM.NAS_START_EVAL_EPOCH -1 \
SLIM.NAS_IS_SERVER True \
SLIM.NAS_SPACE_NAME "MobileNetV2SpaceSeg" \
SOLVER.NUM_EPOCHS 1 \
BATCH_SIZE 7
}
model=seg_nas_train_1card
CUDA_VISIBLE_DEVICES=${cudaid1} nas_train 23332 > ${log_path}/${model} 2>&1
print_info $? ${model}
model=seg_nas_train_8card
CUDA_VISIBLE_DEVICES=${cudaid8} nas_train 23333 > ${log_path}/${model} 2>&1
print_info $? ${model}

# tar models_from_train for lite
cd $(dirname ${models_from_train})
echo $PWD
if [ "$(ls -A ${models_from_train})" ];then
   tar -czf models_from_train.tar.gz models_from_train
   echo "\033[32m -----models_from_train tar SUCCESS----- \033[0m"
else
   echo "\033[31m -----models_from_train is empty----- \033[0m"
fi
