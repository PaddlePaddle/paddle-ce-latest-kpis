#!/bin/bash

rm -rf *_factor.txt
export current_dir=$PWD
#  for lite models path
if [ -d "/ssd2/models_from_train" ];then
	rm -rf /ssd2/models_from_train
fi
mkdir /ssd2/models_from_train
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
#######################################################
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
CUDA_VISIBLE_DEVICES=${cudaid1} dete_dist_yolov3_v1 8000 >dete_dist_yolov3_v1_1card 2>&1
cat dete_dist_yolov3_v1_1card|grep Best | awk -F ' ' 'END{print "kpis\tdete_dist_yolov3_v1_bestap_1card\t"$7}'|tr -d ',' | python _ce.py
CUDA_VISIBLE_DEVICES=${cudaid8} dete_dist_yolov3_v1 2000 >dete_dist_yolov3_v1_8card 2>&1
cat dete_dist_yolov3_v1_8card|grep Best | awk -F ' ' 'END{print "kpis\tdete_dist_yolov3_v1_bestap_8card\t"$7}'|tr -d ',' | python _ce.py

# 1.2 infer
model=dete_dist_yolov3_mobilenet_v1_voc_infer
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u tools/infer.py \
-c configs/yolov3_mobilenet_v1_voc.yml \
--infer_img=demo/000000570688.jpg \
--output_dir=infer_output/ \
--draw_threshold=0.5 >${model} 2>&1
print_info $? ${model}
#1.3 eval
model=dete_dist_yolov3_mobilenet_v1_voc_eval
python -u tools/eval.py -c configs/yolov3_mobilenet_v1_voc.yml >${model} 2>&1
print_info $? ${model}

if [ -d "output" ];then
	mv  output dist_output
fi
# for lite
mkdir dete_dist_yolov3_v1_uncombined
cp dist_output/yolov3_mobilenet_v1_voc/infer/* dete_dist_yolov3_v1_uncombined/
copy_for_lite dete_dist_yolov3_v1_uncombined ${models_from_train}



