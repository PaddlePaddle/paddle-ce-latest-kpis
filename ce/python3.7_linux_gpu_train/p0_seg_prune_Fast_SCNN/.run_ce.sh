#!/bin/bash

rm -rf *_factor.txt
export current_dir=$PWD
#  for lite models path
if [ ! -d "/ssd2/models_from_train" ];then
	mkdir /ssd2/models_from_train
fi
export models_from_train=/ssd2/models_from_train

print_info(){
if [ $1 -ne 0 ];then
    echo -e "\033[31m $2_FAIL \033[0m"
else
    echo -e "\033[32m $2_SUCCESS \033[0m"
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
cudaid2=${card8:=2,3} # use 0-th card as default
####################################################
export PYTHONPATH=$PYTHONPATH:./pdseg

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
CUDA_VISIBLE_DEVICES=${cudaid1} seg_prune_Fast_SCNN 1>seg_prune_Fast_SCNN_1card 2>&1
cat seg_prune_Fast_SCNN_1card |grep image=500 |awk -F ' |=' 'END{print "kpis\tseg_prune_Fast_SCNN_acc_1card\t"$4"\tkpis\tseg_prune_Fast_SCNN_IoU_1card\t"$6}' | python _ce.py
CUDA_VISIBLE_DEVICES=${cudaid8} seg_prune_Fast_SCNN 1>seg_prune_Fast_SCNN_8card 2>&1
cat seg_prune_Fast_SCNN_8card |grep image=500 |awk -F ' |=' 'END{print "kpis\tseg_prune_Fast_SCNN_acc_8card\t"$4"\tkpis\tseg_prune_Fast_SCNN_IoU_8card\t"$6}' | python _ce.py

# 3.2 seg/prune eval
model=seg_prune_Fast_SCNN_eval
CUDA_VISIBLE_DEVICES=${cudaid1} python -u ./slim/prune/eval_prune.py \
--cfg configs/cityscape_fast_scnn.yaml \
--use_gpu \
TEST.TEST_MODEL ./snapshots/cityscape_fast_scnn/final >${model} 2>&1
print_info $? ${model}

# 3.3 prune no infer

# tar models_from_train for lite
cd $(dirname ${models_from_train})
echo $PWD
if [ "$(ls -A ${models_from_train})" ];then
   tar -czf models_from_train.tar.gz models_from_train
   echo "\033[32m -----models_from_train tar SUCCESS----- \033[0m"
else
   echo "\033[31m -----models_from_train is empty----- \033[0m"
fi
