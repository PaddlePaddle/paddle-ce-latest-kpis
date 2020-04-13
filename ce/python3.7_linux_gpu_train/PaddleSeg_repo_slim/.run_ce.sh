#!/bin/bash

rm -rf *_factor.txt
export current_dir=$PWD
if [ ! -d "/ssd2/models_from_train" ];then
	mkdir /ssd2/models_from_train
fi
export models_from_train=/ssd2/models_from_train

print_info()
{
if [ $1 -ne 0 ];then
	echo -e "$2,FAIL"
else
	echo -e "$2,SUCCESS"
fi
}

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
--do_eval
}
CUDA_VISIBLE_DEVICES=7 seg_dist_Dv3_xception_mobilenet 1>seg_dist_Dv3_xception_mobilenet_1card 2>&1
cat seg_dist_Dv3_xception_mobilenet_1card |grep image=500 |awk -F ' |=' 'END{print "kpis\tseg_dist_Dv3_xception_mobilenet_acc_1card\t"$4"\tkpis\tseg_dist_Dv3_xception_mobilenet_IoU_1card\t"$6}' | python _ce.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 seg_dist_Dv3_xception_mobilenet 1>seg_dist_Dv3_xception_mobilenet_8card 2>&1
cat seg_dist_Dv3_xception_mobilenet_8card |grep image=500 |awk -F ' |=' 'END{print "kpis\tseg_dist_Dv3_xception_mobilenet_acc_8card\t"$4"\tkpis\tseg_dist_Dv3_xception_mobilenet_IoU_8card\t"$6}' | python _ce.py

# 1.2  dist export
model=dist_Dv3_xception_mobilenet_export
CUDA_VISIBLE_DEVICES=7 python pdseg/export_model.py --cfg ./slim/distillation/cityscape.yaml \
TEST.TEST_MODEL ./snapshots/cityscape_mbv2_kd_e100_1/final >${model} 2>&1
print_info $? ${model}
#infer
# 1.3  dist infer environment
cd ./deploy/python
pip install -r requirements.txt
apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
model=dist_Dv3_xception_mobilenet_infer
cd ${current_dir}
python ./deploy/python/infer.py --conf=./freeze_model/deploy.yaml \
--input_dir=./test_img --use_pr=True >${model} 2>&1
print_info $? ${model}
mv freeze_model seg_dist_v3_xception_mobilenet
# lite op not support
#cp -r seg_dist_v3_xception_mobilenet  ${models_from_train}/
# 1.4 dist eval
model=dist_Dv3_xception_mobilenet_eval
CUDA_VISIBLE_DEVICES=7 python pdseg/eval.py --use_gpu --cfg ./slim/distillation/cityscape.yaml \
TEST.TEST_MODEL ./snapshots/cityscape_mbv2_kd_e100_1/final >${model} 2>&1
print_info $? ${model}

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
BATCH_SIZE 16
}
CUDA_VISIBLE_DEVICES=7 seg_quan_Deeplabv3_v2 1>seg_quan_Deeplabv3_v2_1card 2>&1
cat seg_quan_Deeplabv3_v2_1card |grep image=500 |awk -F ' |=' 'END{print "kpis\tseg_quan_Dv3_v2_acc_1card\t"$4"\tkpis\tseg_quan_Dv3_v2_IoU_1card\t"$6}' | python _ce.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 seg_quan_Deeplabv3_v2 1>seg_quan_Deeplabv3_v2_8card 2>&1
cat seg_quan_Deeplabv3_v2_8card |grep image=500 |awk -F ' |=' 'END{print "kpis\tseg_quan_Dv3_v2_acc_8card\t"$4"\tkpis\tseg_quan_Dv3_v2_IoU_8card\t"$6}' | python _ce.py

# 2.2  seg quan export  RD代码未合入
#model=seg_quan_Dv3_v2_export
#python -u ./slim/quantization/export_model.py \
#--not_quant_pattern last_conv  \
#--cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml  \
#TEST.TEST_MODEL "./snapshots/mobilenetv2_quant/best_model" \
#MODEL.DEEPLAB.ENCODER_WITH_ASPP False \
#MODEL.DEEPLAB.ENABLE_DECODER False \
#TRAIN.SYNC_BATCH_NORM False \
#SLIM.PREPROCESS True >${model} 2>&1
#print_info $? ${model}

# 2.3 quan infer
# infer environment
#model=seg_quan_Dv3_v2_infer
#python ./deploy/python/infer.py --conf=./freeze_model/deploy.yaml \
#--input_dir=./test_img --use_pr=False >${model}.log 2>&1
#print_info $? ${model}
# 转存
#mv freeze_model quan_Dv3_v2_freeze_model
#cp -r quan_Dv3_v2_freeze_model  ${models_from_train}/

# 2.4 quan_eval
model=seg_quan_Dv3_v2_eval_quant
CUDA_VISIBLE_DEVICES=7 python -u ./slim/quantization/eval_quant.py \
--cfg configs/deeplabv3p_mobilenetv2_cityscapes.yaml  \
--use_gpu --not_quant_pattern last_conv --convert \
TEST.TEST_MODEL "./snapshots/mobilenetv2_quant/best_model" \
MODEL.DEEPLAB.ENCODER_WITH_ASPP False \
MODEL.DEEPLAB.ENABLE_DECODER False \
TRAIN.SYNC_BATCH_NORM False \
BATCH_SIZE 16 >${model} 2>&1
print_info $? ${model}

# 3 prune
# 3.1 prune train
seg_prune_Fast_SCNN (){
python -u ./slim/prune/train_prune.py \
--log_steps 10 \
--cfg configs/cityscape_fast_scnn.yaml \
--use_gpu \
--do_eval \
BATCH_SIZE 16 \
SLIM.PRUNE_PARAMS 'learning_to_downsample/weights,learning_to_downsample/dsconv1/pointwise/weights,learning_to_downsample/dsconv2/pointwise/weights' \
SLIM.PRUNE_RATIOS '[0.1,0.1,0.1]' \
TRAIN.SNAPSHOT_EPOCH 1 \
SOLVER.NUM_EPOCHS 1
}
CUDA_VISIBLE_DEVICES=7 seg_prune_Fast_SCNN 1>seg_prune_Fast_SCNN_1card 2>&1
cat seg_prune_Fast_SCNN_1card |grep image=500 |awk -F ' |=' 'END{print "kpis\tseg_prune_Fast_SCNN_acc_1card\t"$4"\tkpis\tseg_prune_Fast_SCNN_IoU_1card\t"$6}' | python _ce.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 seg_prune_Fast_SCNN 1>seg_prune_Fast_SCNN_8card 2>&1
cat seg_prune_Fast_SCNN_8card |grep image=500 |awk -F ' |=' 'END{print "kpis\tseg_prune_Fast_SCNN_acc_8card\t"$4"\tkpis\tseg_prune_Fast_SCNN_IoU_8card\t"$6}' | python _ce.py

# 3.2 seg/prune eval
model=seg_prune_Fast_SCNN_eval
CUDA_VISIBLE_DEVICES=7 python -u ./slim/prune/eval_prune.py \
--cfg configs/cityscape_fast_scnn.yaml \
--use_gpu \
TEST.TEST_MODEL ./snapshots/cityscape_fast_scnn/final >${model} 2>&1
print_info $? ${model}

# 3.3 prune infer  暂无,RD export还未添加

# 4 nas 无指标未添加

