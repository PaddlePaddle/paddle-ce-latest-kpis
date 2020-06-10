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
cudaid4=${card4:=0,1,2,3} # use 0-th card as default

#————————————————————————————————————————————————
# 1 distillation
cd ${current_dir}/demo/distillation
dist_student=(MobileNet ResNet50 MobileNetV2_x0_25)
dist_teacher=(ResNet50_vd ResNet101_vd MobileNetV2)
batch_size_1card=(256 128 512)
batch_size_8card=(32 32 64)
train_dist(){
python distill.py \
--num_epochs 1 \
--batch_size $3 \
--save_inference True \
--model $1 \
--teacher_model $2 \
--teacher_pretrained_model ../pretrain/$2_pretrained
}
for i in $(seq 0 2); do
    CUDA_VISIBLE_DEVICES=${cudaid1} train_dist ${dist_student[$i]} ${dist_teacher[$i]} ${batch_size_1card[$i]} >${current_dir}/dist_${dist_teacher[$i]}_${dist_student[$i]}_gpu1 2>&1
    tail -20 ${current_dir}/dist_${dist_teacher[$i]}_${dist_student[$i]}_gpu1 > ${current_dir}/dist_${dist_teacher[$i]}_${dist_student[$i]}_gpu1_20
    cd ${current_dir}
    cat ${current_dir}/dist_${dist_teacher[$i]}_${dist_student[$i]}_gpu1_20|grep epoch |grep top1 |tr -d ','|awk -F ' ' 'END{print "kpis\t""'dist_${dist_teacher[$i]}_${dist_student[$i]}_acc_top1_gpu1'""\t"$6"\nkpis\t""'dist_${dist_teacher[$i]}_${dist_student[$i]}_acc_top5_gpu1'""\t"$8}' | python _ce.py
    cd ${current_dir}/demo/distillation
    CUDA_VISIBLE_DEVICES=${cudaid8} train_dist ${dist_student[$i]} ${dist_teacher[$i]} ${batch_size_8card[$i]} >${current_dir}/dist_${dist_teacher[$i]}_${dist_student[$i]}_gpu8 2>&1
    tail -20 ${current_dir}/dist_${dist_teacher[$i]}_${dist_student[$i]}_gpu8 > ${current_dir}/dist_${dist_teacher[$i]}_${dist_student[$i]}_gpu8_20
    cd ${current_dir}
    cat ${current_dir}/dist_${dist_teacher[$i]}_${dist_student[$i]}_gpu8_20|grep epoch |grep top1 |tr -d ','|awk -F ' ' 'END{print "kpis\t""'dist_${dist_teacher[$i]}_${dist_student[$i]}_acc_top1_gpu8'""\t"$6"\nkpis\t""'dist_${dist_teacher[$i]}_${dist_student[$i]}_acc_top5_gpu8'""\t"$8}' | python _ce.py
    #move models for lite
    cd ${current_dir}/demo/distillation
    mkdir slim_dist_${dist_teacher[$i]}_${dist_student[$i]}_uncombined
    cp ./saved_models/0/* slim_dist_${dist_teacher[$i]}_${dist_student[$i]}_uncombined/
    copy_for_lite slim_dist_${dist_teacher[$i]}_${dist_student[$i]}_uncombined ${models_from_train}
    if [ -d "saved_models" ];then
	    mv  saved_models ${dist_teacher[$i]}_${dist_student[$i]}_saved_models
    fi
done

# 2.1 quant/quant_aware
cd ${current_dir}/demo/quant/quant_aware
quan_aware_models=(ResNet34)
quan_aware_train(){
python train.py \
--model $1 \
--pretrained_model ../../pretrain/$1_pretrained \
--checkpoint_dir ./output/$1 \
--num_epochs 1
}
for i in $(seq 0 0); do
    CUDA_VISIBLE_DEVICES=${cudaid1} quan_aware_train ${quan_aware_models[$i]} >${current_dir}/quan_aware_${quan_aware_models[$i]}_gpu1 2>&1
    cd ${current_dir}
    cat ${current_dir}/quan_aware_${quan_aware_models[$i]}_gpu1|grep Final |awk -F ' ' 'END{print "kpis\tquant_aware_""'${quan_aware_models[$i]}_acc_top1_gpu1'""\t"$8"\nkpis\tquant_aware_""'${quan_aware_models[$i]}_acc_top5_gpu1'""\t"$10}'|tr -d ";" | python _ce.py
    cd ${current_dir}/demo/quant/quant_aware
    CUDA_VISIBLE_DEVICES=${cudaid8} quan_aware_train ${quan_aware_models[$i]} >${current_dir}/quan_aware_${quan_aware_models[$i]}_gpu8 2>&1
    cd ${current_dir}
    cat ${current_dir}/quan_aware_${quan_aware_models[$i]}_gpu8|grep Final |awk -F ' ' 'END{print "kpis\tquant_aware_""'${quan_aware_models[$i]}_acc_top1_gpu8'""\t"$8"\nkpis\tquant_aware_""'${quan_aware_models[$i]}_acc_top5_gpu8'""\t"$10}'|tr -d ';' | python _ce.py
    cd ${current_dir}/demo/quant/quant_aware
#    for lite
    mkdir slim_quan_aware_${quan_aware_models[$i]}_combined
    cp ./quantization_models/${quan_aware_models[$i]}/act_moving_average_abs_max_w_channel_wise_abs_max/float/* ./slim_quan_aware_${quan_aware_models[$i]}_combined/
    mv ./slim_quan_aware_${quan_aware_models[$i]}_combined/model ./slim_quan_aware_${quan_aware_models[$i]}_combined/__model__
    mv ./slim_quan_aware_${quan_aware_models[$i]}_combined/params ./slim_quan_aware_${quan_aware_models[$i]}_combined/__params__
    copy_for_lite slim_quan_aware_${quan_aware_models[$i]}_combined ${models_from_train}
    if [ -d "models" ];then
	    mv  models ${quan_aware_models[$i]}
    fi
done
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
# 2.2 quant/quant_embedding
cd ${current_dir}/demo/quant/quant_embedding
if [ -d "data" ];then
    rm -rf data
fi

ln -s  ${dataset_path}/word2vec/demo_data data
if [ -d "v1_cpu5_b100_lr1dir" ];then
    rm -rf v1_cpu5_b100_lr1dir
fi
model=quant_em_word2vec_T
time (OPENBLAS_NUM_THREADS=1 CPU_NUM=5 python train.py --train_data_dir data/convert_text8 --dict_path data/test_build_dict --num_passes 1 --batch_size 100 --model_output_dir v1_cpu5_b100_lr1dir --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
# before quan infer
model=quant_em_infer1
time (python infer.py --infer_epoch --test_dir data/test_mid_dir --dict_path data/test_build_dict_word_to_id_ --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 --last_index 0 >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
# after quan infer
model=quant_em_infer2
time (python infer.py --infer_epoch --test_dir data/test_mid_dir --dict_path data/test_build_dict_word_to_id_ --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 --last_index 0 --emb_quant True >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}

# 2.3 quan_post
cd ${current_dir}/demo/quant/quant_post
# export for quan
model=quant_post_export
CUDA_VISIBLE_DEVICES=${cudaid1} python export_model.py --model "MobileNet" --pretrained_model ../../pretrain/MobileNetV1_pretrained --data imagenet >${log_path}/${model} 2>&1
print_info $? ${model}
#before quan ;inference_model/  combined
mkdir slim_quan_MobileNet_post_1_combined
cp ./inference_model/MobileNet/* ./slim_quan_MobileNet_post_1_combined/
mv ./slim_quan_MobileNet_post_1_combined/model ./slim_quan_MobileNet_post_1_combined/__model__
mv ./slim_quan_MobileNet_post_1_combined/weights ./slim_quan_MobileNet_post_1_combined/__params__
copy_for_lite slim_quan_MobileNet_post_1_combined ${models_from_train}/
# quant_post
model=quant_post_T
CUDA_VISIBLE_DEVICES=${cudaid1} python quant_post.py --model_path ./inference_model/MobileNet --save_path ./quant_model_train/MobileNet --model_filename model --params_filename weights >${log_path}/${model} 2>&1
print_info $? ${model}
# before quan eval
model=quant_post_eval1
CUDA_VISIBLE_DEVICES=${cudaid1} python eval.py --model_path ./inference_model/MobileNet --model_name model --params_name weights >${log_path}/${model} 2>&1
print_info $? ${model}
# after quan eval
model=quant_post_eval2
CUDA_VISIBLE_DEVICES=${cudaid1} python eval.py --model_path ./quant_model_train/MobileNet --model_name __model__ --params_name __params__ >${log_path}/${model} 2>&1
print_info $? ${model}
#for lite combined
mkdir slim_quan_MobileNet_post_2_combined
cp ./quant_model_train/MobileNet/* ./slim_quan_MobileNet_post_2_combined/
copy_for_lite slim_quan_MobileNet_post_2_combined ${models_from_train}

#3.1 prune MobileNetV1
cd ${current_dir}/demo/prune
CUDA_VISIBLE_DEVICES=${cudaid1} python train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" --pretrained_model ../pretrain/MobileNetV1_pretrained/ --num_epochs 1 --save_inference True >${current_dir}/prune_v1_T_1card 2>&1
cd ${current_dir}
cat prune_v1_T_1card |grep Final |awk -F ' ' 'END{print "kpis\tprune_v1_acc_top1_gpu1\t"$8"\nkpis\tprune_v1_acc_top5_gpu1\t"$10}' |tr -d ";" | python _ce.py
cd ${current_dir}/demo/prune
CUDA_VISIBLE_DEVICES=${cudaid8} python train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" --pretrained_model ./MobileNetV1_pretrained/ --num_epochs 1  --save_inference True >${current_dir}/prune_v1_T_8card 2>&1
# for lite uncombined
mkdir slim_prune_MobileNetv1_uncombined
cp ./models/infer_models/0/* ./slim_prune_MobileNetv1_uncombined/
copy_for_lite slim_prune_MobileNetv1_uncombined ${models_from_train}
cd ${current_dir}
cat prune_v1_T_8card |grep Final |awk -F ' ' 'END{print "kpis\tprune_v1_acc_top1_gpu8\t"$8"\nkpis\tprune_v1_acc_top5_gpu8\t"$10}' |tr -d ";" | python _ce.py
# 3.2 prune eval
cd ${current_dir}/demo/prune
model=slim_prune_eval
python eval.py --model "MobileNet" --data "imagenet" --model_path "./models/0"  >${log_path}/${model} 2>&1
print_info $? ${model}
if [ -d "models" ];then
    mv  models MobileNet_models
fi

#3.2 prune_fpgm
cd ${current_dir}/demo/prune
slim_prune_fpgm_v1 (){
python train.py \
    --model="MobileNet" \
    --pretrained_model="../pretrain/MobileNetV1_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.3125 \
    --lr=0.1 \
    --num_epochs=1 \
    --test_period=1 \
    --step_epochs 30 60 90\
    --l2_decay=3e-5 \
    --lr_strategy="piecewise_decay" \
    --criterion="geometry_median" \
    --model_path="./fpgm_mobilenetv1_models" \
    --save_inference True
}
CUDA_VISIBLE_DEVICES=${cudaid1} slim_prune_fpgm_v1 >${current_dir}/slim_prune_fpgm_v1_f50_T_1card 2>&1
cd ${current_dir}
cat slim_prune_fpgm_v1_f50_T_1card |grep Final |awk -F ' ' 'END{print "kpis\tprune_fpgm_v1_f50_acc_top1_gpu1\t"$8"\nkpis\tprune_fpgm_v1_f50_acc_top5_gpu1\t"$10}' |tr -d ";" | python _ce.py
cd ${current_dir}/demo/prune
CUDA_VISIBLE_DEVICES=${cudaid8} slim_prune_fpgm_v1 >${current_dir}/slim_prune_fpgm_v1_f50_T_8card 2>&1
# for lite uncombined
mkdir slim_prune_fpgm_v1_f50_uncombined
cp ./fpgm_mobilenetv1_models/infer_models/0/* ./slim_prune_fpgm_v1_f50_uncombined/
copy_for_lite slim_prune_fpgm_v1_f50_uncombined ${models_from_train}
cd ${current_dir}
cat slim_prune_fpgm_v1_f50_T_8card |grep Final |awk -F ' ' 'END{print "kpis\tprune_fpgm_v1_f50_acc_top1_gpu8\t"$8"\nkpis\tprune_fpgm_v1_f50_acc_top5_gpu8\t"$10}' |tr -d ";" | python _ce.py
# 3.2.2 prune eval
cd ${current_dir}/demo/prune
model=slim_prune_fpgm_v1_eval
python eval.py --model "MobileNet" --data "imagenet" --model_path "./fpgm_mobilenetv1_models/0"  >${log_path}/${model} 2>&1
print_info $? ${model}

cd ${current_dir}/demo/prune
slim_prune_fpgm_resnet34(){
python train.py \
    --model="ResNet34" \
    --pretrained_model="../pretrain/ResNet34_pretrained" \
    --data="imagenet" \
    --pruned_ratio=0.3125 \
    --lr=0.001 \
    --num_epochs=1 \
    --test_period=1 \
    --step_epochs 30 60 \
    --l2_decay=1e-4 \
    --lr_strategy="piecewise_decay" \
    --criterion="geometry_median" \
    --model_path="./fpgm_resnet34_models" \
    --save_inference True
}
CUDA_VISIBLE_DEVICES=${cudaid1} slim_prune_fpgm_resnet34 >${current_dir}/slim_prune_fpgm_resnet34_f50_T_1card 2>&1
cd ${current_dir}
cat slim_prune_fpgm_resnet34_f50_T_1card |grep Final |awk -F ' ' 'END{print "kpis\tprune_fpgm_resnet34_f50_acc_top1_gpu1\t"$8"\nkpis\tprune_fpgm_resnet34_f50_acc_top5_gpu1\t"$10}' |tr -d ";" | python _ce.py
cd ${current_dir}/demo/prune
CUDA_VISIBLE_DEVICES=${cudaid8} slim_prune_fpgm_v1 >${current_dir}/slim_prune_fpgm_resnet34_f50_T_8card 2>&1
# for lite uncombined
mkdir slim_prune_fpgm_resnet34_f50_uncombined
cp ./fpgm_resnet34_models/infer_models/0/* ./slim_prune_fpgm_resnet34_f50_uncombined/
copy_for_lite slim_prune_fpgm_resnet34_f50_uncombined ${models_from_train}
cd ${current_dir}
cat slim_prune_fpgm_resnet34_f50_T_8card |grep Final |awk -F ' ' 'END{print "kpis\tprune_fpgm_resnet34_f50_acc_top1_gpu8\t"$8"\nkpis\tprune_fpgm_resnet34_f50_acc_top5_gpu8\t"$10}' |tr -d ";" | python _ce.py
# 3.2.2 prune eval
cd ${current_dir}/demo/prune
model=slim_prune_fpgm_resnet34_eval
python eval.py --model "ResNet34" --data "imagenet" --model_path "./fpgm_resnet34_models/0"  >${log_path}/${model} 2>&1
print_info $? ${model}

# 3.3 prune ResNet50
cd ${current_dir}/demo/prune
prune_models=(ResNet50)
train_prune(){
python train.py \
--model $1 \
--pruned_ratio 0.31 \
--data "imagenet" \
--save_inference True \
--pretrained_model ../pretrain/$1_pretrained/ \
--num_epochs 1
}

eval_prune(){
    python eval.py --model $1 --data "imagenet" --model_path "./models/0"
}
for i in $(seq 0 0); do
    CUDA_VISIBLE_DEVICES=${cudaid1} train_prune ${prune_models[$i]} >${current_dir}/prune_${prune_models[$i]}_gpu1 2>&1
    cd ${current_dir}
    cat ${current_dir}/prune_${prune_models[$i]}_gpu1|grep Final |awk -F ' ' 'END{print "kpis\tprune_""'${prune_models[$i]}_acc_top1_gpu1'""\t"$8"\nkpis\tprune_""'${prune_models[$i]}_acc_top5_gpu1'""\t"$10}'|tr -d ';' | python _ce.py
    cd ${current_dir}/demo/prune
    CUDA_VISIBLE_DEVICES=${cudaid8} train_prune ${prune_models[$i]} >${current_dir}/prune_${prune_models[$i]}_gpu8 2>&1
    cd ${current_dir}
    cat ${current_dir}/prune_${prune_models[$i]}_gpu8|grep Final |awk -F ' ' 'END{print "kpis\tprune_""'${prune_models[$i]}_acc_top1_gpu8'""\t"$8"\nkpis\tprune_""'${prune_models[$i]}_acc_top5_gpu8'""\t"$10}'|tr -d ';' | python _ce.py
    #move models for lite uncombined
    cd ${current_dir}/demo/prune
    mkdir slim_prune_${prune_models[$i]}_uncombined
    cp ./models/infer_models/0/* slim_prune_${prune_models[$i]}_uncombined/
    copy_for_lite slim_prune_${prune_models[$i]}_uncombined ${models_from_train}
    eval_prune ${prune_models[$i]} >prune_${prune_models[$i]}_eval 2>&1
    print_info $? prune_${prune_models[$i]}_eval
    if [ -d "models" ];then
	    mv  models ${prune_models[$i]}_models
    fi
done
#4 nas
# 4.1 sa_nas_mobilenetv2
cd ${current_dir}/demo/nas
model=sa_nas_v2_T_1card
time (CUDA_VISIBLE_DEVICES=${cudaid1} python sa_nas_mobilenetv2.py --search_steps 1 --port 8881 >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
model=sa_nas_v2_T_8card
time (CUDA_VISIBLE_DEVICES=${cudaid8} python sa_nas_mobilenetv2.py --search_steps 1 --port 8882 >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}

# 4.2 block_sa_nas_mobilenetv2
model=block_sa_nas_v2_T_1card
time (CUDA_VISIBLE_DEVICES=${cudaid1} python block_sa_nas_mobilenetv2.py --search_steps 1 --port 8883 >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
model=block_sa_nas_v2_T_8card
time (CUDA_VISIBLE_DEVICES=${cudaid8} python block_sa_nas_mobilenetv2.py --search_steps 1 --port 8884 >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}

# 4.3 rl_nas
model=rl_nas_v2_T_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python rl_nas_mobilenetv2.py --search_steps 1 --port 8885 >${log_path}/${model} 2>&1
print_info $? ${model}
model=rl_nas_v2_T_8card
CUDA_VISIBLE_DEVICES=${cudaid8} python rl_nas_mobilenetv2.py --search_steps 1 --port 8886 >${log_path}/${model} 2>&1
print_info $? ${model}

# 4.4 parl_nas
model=parl_nas_v2_T_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python parl_nas_mobilenetv2.py --search_steps 1 --port 8887 >${log_path}/${model} 2>&1
print_info $? ${model}
model=parl_nas_v2_T_8card
CUDA_VISIBLE_DEVICES=${cudaid8} python parl_nas_mobilenetv2.py --search_steps 1 --port 8889 >${log_path}/${model} 2>&1
print_info $? ${model}

# 5 darts
# search 1card # DARTS一阶近似搜索方法
cd ${current_dir}/demo/darts
model=darts1_search_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python search.py --epochs 1 --use_multiprocess False >${log_path}/${model} 2>&1
print_info $? ${model}
model=darts1_search_8card
CUDA_VISIBLE_DEVICES=${cudaid8} python search.py --epochs 1 --use_multiprocess False >${log_path}/${model} 2>&1
print_info $? ${model}
# # DARTS 二阶近似搜索方法
model=darts2_search_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python search.py --epochs 1 --unrolled=True --use_multiprocess False >${log_path}/${model} 2>&1
print_info $? ${model}
model=darts2_search_8card
CUDA_VISIBLE_DEVICES=${cudaid8} python search.py --epochs 1 --unrolled=True --use_multiprocess False >${log_path}/${model} 2>&1
print_info $? ${model}
# PC-DARTS
model=pcdarts_search_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python search.py --epochs 1 --method='PC-DARTS' --use_multiprocess False --batch_size=256 --learning_rate=0.1 --arch_learning_rate=6e-4 --epochs_no_archopt=15 >${log_path}/${model} 2>&1
print_info $? ${model}
model=pcdarts_search_8card
CUDA_VISIBLE_DEVICES=${cudaid8} python search.py --epochs 1 --method='PC-DARTS' --use_multiprocess False --batch_size=256 --learning_rate=0.1 --arch_learning_rate=6e-4 --epochs_no_archopt=15 >${log_path}/${model} 2>&1
print_info $? ${model}
# 分布式 search
model=darts1_search_distributed
CUDA_VISIBLE_DEVICES=${cudaid4} python -m paddle.distributed.launch --selected_gpus=0,1,2,3 --log_dir ./mylog_search  search.py --use_data_parallel 1 --epochs 1 --use_multiprocess False >${log_path}/${model} 2>&1
print_info $? ${model}
model=darts2_search_distributed
CUDA_VISIBLE_DEVICES=${cudaid4} python -m paddle.distributed.launch --selected_gpus=0,1,2,3 --log_dir ./mylog_search  search.py --use_data_parallel 1 --epochs 1 --unrolled=True --use_multiprocess False >${log_path}/${model} 2>&1
print_info $? ${model}
model=pcdarts_search_distributed
CUDA_VISIBLE_DEVICES=${cudaid4} python -m paddle.distributed.launch --selected_gpus=0,1,2,3 --log_dir ./mylog_search  search.py --use_data_parallel 1 --epochs 1 --use_multiprocess False --epochs 1 --method='PC-DARTS' --use_multiprocess False --batch_size=256 --learning_rate=0.1 --arch_learning_rate=6e-4 --epochs_no_archopt=15 >${log_path}/${model} 2>&1
print_info $? ${model}
#train
model=pcdarts_train_1card
CUDA_VISIBLE_DEVICES=${cudaid1} python train.py --arch='PC_DARTS' --epochs 1 --use_multiprocess False >${log_path}/${model} 2>&1
print_info $? ${model}
model=pcdarts_train_imagenet_8card
CUDA_VISIBLE_DEVICES=${cudaid8} python train_imagenet.py --arch='PC_DARTS' --epochs 1 --use_multiprocess False --data_dir ../data/ILSVRC2012 >${log_path}/${model} 2>&1
print_info $? ${model}
# 分布式 train
model=dartsv2_train_distributed
CUDA_VISIBLE_DEVICES=${cudaid4} python -m paddle.distributed.launch --selected_gpus=0,1,2,3 --log_dir ./mylog_train train.py --use_data_parallel 1 --arch='DARTS_V2' >${log_path}/${model} 2>&1
print_info $? ${model}
model=dartsv2_train_imagenet_distributed
CUDA_VISIBLE_DEVICES=${cudaid4} python -m paddle.distributed.launch --selected_gpus=0,1,2,3 --log_dir ./mylog_train_imagenet train_imagenet.py --use_data_parallel 1 --arch='DARTS_V2' --data_dir ../data/ILSVRC2012 >${log_path}/${model} 2>&1
print_info $? ${model}
# 可视化
yum -y install graphviz
model=slim_darts_visualize_pcdarts
python visualize.py PC_DARTS > ${log_path}/${model} 2>&1
print_info $? ${model}

