#!/bin/bash
rm -rf *_factor.txt
export current_dir=$PWD
#  for lite models path
if [ ! -d "/ssd2/models_from_train" ];then
	mkdir /ssd2/models_from_train
fi
export models_from_train=/ssd2/models_from_train

#set log dir
cd ${current_dir}
if [ -d "ce_logs" ];then
    rm -rf ce_logs
fi
mkdir ce_logs && cd ce_logs
mkdir SUCCESS
mkdir FAIL
log_path=${current_dir}"/ce_logs"
print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/$2_FAIL
    echo -e "\033[31m ${log_path}/$2_FAIL \033[0m"
else
    mv ${log_path}/$2 ${log_path}/$2_SUCCESS
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

#————————————————————————————————————————————————

# 2.1 quant/quant_aware
# quant_aware MobileNet
cd ${current_dir}/demo/quant/quant_aware
CUDA_VISIBLE_DEVICES=${cudaid1} python train.py --model MobileNet --pretrained_model ../../pretrain/MobileNetV1_pretrained --checkpoint_dir ./output/mobilenetv1 --num_epochs 1 >${current_dir}/quant_aware_v1_1card 2>&1
cd ${current_dir}
cat quant_aware_v1_1card |grep Final |awk -F ' ' 'END{print "kpis\tquant_aware_v1_acc_top1_gpu1\t"$8"\nkpis\tquant_aware_v1_acc_top5_gpu1\t"$10}' |tr -d ";" | python _ce.py
cd ${current_dir}/demo/quant/quant_aware
CUDA_VISIBLE_DEVICES=${cudaid8} python train.py --model MobileNet --pretrained_model ../../pretrain/MobileNetV1_pretrained --checkpoint_dir ./output/mobilenetv1 --num_epochs 1 >${current_dir}/quant_aware_v1_8card 2>&1
cd ${current_dir}
cat quant_aware_v1_8card |grep Final |awk -F ' ' 'END{print "kpis\tquant_aware_v1_acc_top1_gpu8\t"$8"\nkpis\tquant_aware_v1_acc_top5_gpu8\t"$10}' |tr -d ";" | python _ce.py
# quantization_models
cd ${current_dir}/demo/quant/quant_aware
mkdir slim_quan_v1_aware_combined
cp ./quantization_models/MobileNet/act_moving_average_abs_max_w_channel_wise_abs_max/float/* ./slim_quan_v1_aware_combined/
mv ./slim_quan_v1_aware_combined/model ./slim_quan_v1_aware_combined/__model__
mv ./slim_quan_v1_aware_combined/params ./slim_quan_v1_aware_combined/__params__
#for lite
copy_for_lite slim_quan_v1_aware_combined ${models_from_train}
if [ -d "models" ];then
    mv  models slim_quan_v1_aware_models
fi
