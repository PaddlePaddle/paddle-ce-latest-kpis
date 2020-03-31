#!/bin/bash

rm -rf *_factor.txt
export current_dir=$PWD
#  for lite models path
if [ ! -d "/ssd2/guomengmeng01/slim/models_from_train" ];then
	mkdir /ssd2/guomengmeng01/slim/models_from_train
fi
export models_from_train=/ssd2/guomengmeng01/slim/models_from_train

#set result dir___________________________________
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
if [ -d "ce_logs" ];then
    rm -rf ce_logs
fi
mkdir ce_logs && cd ce_logs
mkdir SUCCESS
mkdir FAIL
log_path=${current_dir}"/ce_logs"
print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/FAIL/$2
	echo -e "$2,train,FAIL" >>${result_path}/result.log;
else
    mv ${log_path}/$2 ${log_path}/SUCCESS/$2
	echo -e "$2,train,SUCCESS" >>${result_path}/result.log
fi
}
#————————————————————————————————————————————————
cd ${current_dir}/demo/distillation
# 1 distillation
# 1card
CUDA_VISIBLE_DEVICES=0 python distill.py --num_epochs 1 --batch_size 256 --save_inference true >${current_dir}/dist_1card 2>&1
cd ${current_dir}
tail -20 dist_1card > dist_1card_20
cat dist_1card_20 |grep epoch |grep top1 |tr -d ','|awk -F ' ' 'END{print "kpis\tdist_acc_top1_gpu1\t"$6"\nkpis\tdist_acc_top5_gpu1\t"$8}'| python _ce.py

# 8card
cd ${current_dir}/demo/distillation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python distill.py --num_epochs 1 --batch_size 32 --save_inference true >>${current_dir}/dist_8card 2>&1
# 转存路径./saved_models/0
mkdir slim_dist_ResNet50_vd_MobileNet
cp ./saved_models/0/* slim_dist_ResNet50_vd_MobileNet/
cp -r slim_dist_ResNet50_vd_MobileNet ${models_from_train}/
cd ${current_dir}
tail -20 dist_8card > dist_8card_20
cat dist_8card_20 |grep epoch |grep top1 |tr -d ','|awk -F ' ' 'END{print "kpis\tdist_acc_top1_gpu8\t"$6"\nkpis\tdist_acc_top5_gpu8\t"$8}'| python _ce.py

# 2.1 quant/quant_aware
cd ${current_dir}/demo/quant/quant_aware
mkdir pretrain
cd pretrain
wget http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar
tar xf MobileNetV1_pretrained.tar
cd -
CUDA_VISIBLE_DEVICES=0 python train.py --model MobileNet --pretrained_model ./pretrain/MobileNetV1_pretrained --checkpoint_dir ./output/mobilenetv1 --num_epochs 1 >${current_dir}/quant_aware_1card 2>&1
cd ${current_dir}
cat quant_aware_1card |grep Final |awk -F ' ' 'END{print "kpis\tquant_aware_acc_top1_gpu1\t"$8"\nkpis\tquant_aware_acc_top5_gpu1\t"$10}' |tr -d ";" | python _ce.py

cd ${current_dir}/demo/quant/quant_aware
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --model MobileNet --pretrained_model ./pretrain/MobileNetV1_pretrained --checkpoint_dir ./output/mobilenetv1 --num_epochs 1 >${current_dir}/quant_aware_8card 2>&1
cd ${current_dir}
cat quant_aware_8card |grep Final |awk -F ' ' 'END{print "kpis\tquant_aware_acc_top1_gpu8\t"$8"\nkpis\tquant_aware_acc_top5_gpu8\t"$10}' |tr -d ";" | python _ce.py
#导出模型路径quantization_models
cd ${current_dir}/demo/quant/quant_aware
mkdir slim_quan_v1_aware
cp ./quantization_models/MobileNet/act_moving_average_abs_max_w_channel_wise_abs_max/float/* ./slim_quan_v1_aware/
mv ./slim_quan_v1_aware/model ./slim_quan_v1_aware/__model__
mv ./slim_quan_v1_aware/params ./slim_quan_v1_aware/__params__
cp -r ./slim_quan_v1_aware ${models_from_train}/

# 2.2 quant/quant_embedding
cd ${current_dir}/demo/quant/quant_embedding
# 先使用word2vec的demo数据进行一轮训练，比较量化前infer结果同量化后infer结果different
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
# 量化前infer
model=quant_em_infer1
time (python infer.py --infer_epoch --test_dir data/test_mid_dir --dict_path data/test_build_dict_word_to_id_ --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 --last_index 0 >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
# 量化后infer
model=quant_em_infer2
time (python infer.py --infer_epoch --test_dir data/test_mid_dir --dict_path data/test_build_dict_word_to_id_ --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 --last_index 0 --emb_quant True >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}

# 2.3 quan_post
cd ${current_dir}/demo/quant/quant_post
# 导出模型
mkdir pretrain
cd pretrain
wget http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar
tar xf MobileNetV1_pretrained.tar
cd -
model=quant_post_export
time (python export_model.py --model "MobileNet" --pretrained_model ./pretrain/MobileNetV1_pretrained --data imagenet >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
#量化前 导出模型路径inference_model/
mkdir slim_quan_MobileNet_post_1
cp ./inference_model/MobileNet/* ./slim_quan_MobileNet_post_1/
mv ./slim_quan_MobileNet_post_1/model ./slim_quan_MobileNet_post_1/__model__
mv ./slim_quan_MobileNet_post_1/weights ./slim_quan_MobileNet_post_1/__params__
cp -r ./slim_quan_MobileNet_post_1 ${models_from_train}/

#离线量化
model=quant_post_T
time (python quant_post.py --model_path ./inference_model/MobileNet --save_path ./quant_model_train/MobileNet --model_filename model --params_filename weights >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
# 测试精度
# 量化前eval
model=quant_post_eval1
time (python eval.py --model_path ./inference_model/MobileNet --model_name model --params_name weights >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
# 量化后eval
model=quant_post_eval2
time (python eval.py --model_path ./quant_model_train/MobileNet >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
# 量化后 导出模型路径./quant_model_train/MobileNet
mkdir slim_quan_MobileNet_post_2
cp ./quant_model_train/MobileNet/* ./slim_quan_MobileNet_post_2/
cp -r ./slim_quan_MobileNet_post_2 ${models_from_train}/

# 3 prune
cd ${current_dir}/demo/prune
wget http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar
tar -xf MobileNetV1_pretrained.tar
if [ -d "models" ];then
    rm -rf models
fi
CUDA_VISIBLE_DEVICES=3 python train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" --pretrained_model ./MobileNetV1_pretrained/ --num_epochs 1 --save_inference True >${current_dir}/prune_v1_T_1card 2>&1
cd ${current_dir}
cat prune_v1_T_1card |grep Final |awk -F ' ' 'END{print "kpis\tprune_v1_acc_top1_gpu1\t"$8"\nkpis\tprune_v1_acc_top5_gpu1\t"$10}' |tr -d ";" | python _ce.py

cd ${current_dir}/demo/prune
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" --pretrained_model ./MobileNetV1_pretrained/ --num_epochs 1  --save_inference True >${current_dir}/prune_v1_T_8card 2>&1
# 转存
mkdir slim_prune_MobileNetv1
cp ./models/infer_models/0/* ./slim_prune_MobileNetv1/
cp -r ./slim_prune_MobileNetv1 ${models_from_train}/
cd ${current_dir}
cat prune_v1_T_8card |grep Final |awk -F ' ' 'END{print "kpis\tprune_v1_acc_top1_gpu8\t"$8"\nkpis\tprune_v1_acc_top5_gpu8\t"$10}' |tr -d ";" | python _ce.py


#4 nas
# 4.1 sa_nas_mobilenetv2
cd ${current_dir}/demo/nas
model=sa_nas_v2_T_1card
time (CUDA_VISIBLE_DEVICES=3 python sa_nas_mobilenetv2.py --search_steps 1 --port 8881 >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
model=sa_nas_v2_T_8card
time (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python sa_nas_mobilenetv2.py --search_steps 1 --port 8882 >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}

# 4.2 block_sa_nas_mobilenetv2
model=block_sa_nas_v2_T_1card
time (CUDA_VISIBLE_DEVICES=3 python block_sa_nas_mobilenetv2.py --search_steps 1 --port 8883 >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
model=block_sa_nas_v2_T_8card
time (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python block_sa_nas_mobilenetv2.py --search_steps 1 --port 8884 >${log_path}/${model}) >>${log_path}/${model} 2>&1
print_info $? ${model}
