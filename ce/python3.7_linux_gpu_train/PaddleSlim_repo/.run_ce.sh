#!/bin/bash

rm -rf *_factor.txt
# current_dir =/PaddleSlim/
export current_dir=$PWD

#set result dir___________________________________
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
if [ -d "ce_logs" ];then
    rm -rf ce_logs
fi
mkdir ce_logs && cd ce_logs
mkdir SUCCESS
mkdir FAIL
log_path=${current_dir}"/ce_logs"
#————————————————————————————————————————————————
cd ${current_dir}/demo/distillation
# 1 distillation
# 1card
CUDA_VISIBLE_DEVICES=0 python distill.py --num_epochs 1 --batch_size 256 >${current_dir}/dist_1card 2>&1
cd ${current_dir}
tail -20 dist_1card > dist_1card_20
cat dist_1card_20 |grep epoch |grep top1 |tr -d ','|awk -F ' ' 'END{print "kpis\tdist_acc_top1_gpu1\t"$6"\nkpis\tdist_acc_top5_gpu1\t"$8}'| python _ce.py

# 8card
cd ${current_dir}/demo/distillation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python distill.py --num_epochs 1 --batch_size 32 >>${current_dir}/dist_8card 2>&1
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

# 2.2 quant/quant_embedding
cd ${current_dir}/demo/quant/quant_embedding
# 先使用word2vec的demo数据进行一轮训练，比较量化前infer结果同量化后infer结果different
if [ -d "data" ];then
    rm -rf data
fi
# 需要在11 机器上放置demo数据,并在run.xsh中添加软链
#ln -s ${dataset_path}/rec/word2vec/demo_data data
if [ -d "v1_cpu5_b100_lr1dir" ];then
    rm -rf v1_cpu5_b100_lr1dir
fi
time (OPENBLAS_NUM_THREADS=1 CPU_NUM=5 python train.py --train_data_dir data/convert_text8 --dict_path data/test_build_dict --num_passes 1 --batch_size 100 --model_output_dir v1_cpu5_b100_lr1dir --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse >${log_path}/quant_em_word2vec_T.log) >>${log_path}/quant_em_word2vec_T.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/quant_em_word2vec_T.log ${log_path}/FAIL/quant_em_word2vec_T.log
	echo -e "quant_em_word2vec_T,train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/quant_em_word2vec_T.log  ${log_path}/SUCCESS/quant_em_word2vec_T.log
	echo -e "quant_em_word2vec_T,train,SUCCESS" >>${result_path}/result.log
fi
# 量化前infer
time (python infer.py --infer_epoch --test_dir data/test_mid_dir --dict_path data/test_build_dict_word_to_id_ --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 --last_index 0 >${log_path}/quant_em_infer1.log) >>${log_path}/quant_em_infer1.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/quant_em_infer1.log ${log_path}/FAIL/quant_em_infer1.log
	echo -e "quant_em_infer1,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/quant_em_infer1.log  ${log_path}/SUCCESS/quant_em_infer1.log
	echo -e "quant_em_infer1,SUCCESS" >>${result_path}/result.log
fi
# 量化后infer
time (python infer.py --infer_epoch --test_dir data/test_mid_dir --dict_path data/test_build_dict_word_to_id_ --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 --last_index 0 --emb_quant True >${log_path}/quant_em_infer2.log) >>${log_path}/quant_em_infer2.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/quant_em_infer2.log ${log_path}/FAIL/quant_em_infer2.log
	echo -e "quant_em_infer2,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/quant_em_infer2.log  ${log_path}/SUCCESS/quant_em_infer2.log
	echo -e "quant_em_infer2,SUCCESS" >>${result_path}/result.log
fi

# 2.3 quan_post
cd ${current_dir}/demo/quant/quant_post
# 导出模型
mkdir pretrain
cd pretrain
wget http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar
tar xf MobileNetV1_pretrained.tar
cd -
time (python export_model.py --model "MobileNet" --pretrained_model ./pretrain/MobileNetV1_pretrained --data imagenet >${log_path}/quant_post_export.log) >>${log_path}/quant_post_export.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/quant_post_export.log ${log_path}/FAIL/quant_post_export.log
	echo -e "quant_post_export,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/quant_post_export.log  ${log_path}/SUCCESS/quant_post_export.log
	echo -e "quant_post_export,SUCCESS" >>${result_path}/result.log
fi
#离线量化
time (python quant_post.py --model_path ./inference_model/MobileNet --save_path ./quant_model_train/MobileNet --model_filename model --params_filename weights >${log_path}/quant_post_T.log) >>${log_path}/quant_post_T.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/quant_post_T.log ${log_path}/FAIL/quant_post_T.log
	echo -e "quant_post_T,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/quant_post_T.log  ${log_path}/SUCCESS/quant_post_T.log
	echo -e "quant_post_T,SUCCESS" >>${result_path}/result.log
fi
# 测试精度
# 量化前eval
time (python eval.py --model_path ./inference_model/MobileNet --model_name model --params_name weights >${log_path}/quant_post_eval1.log) >>${log_path}/quant_post_eval1.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/quant_post_eval1.log ${log_path}/FAIL/quant_post_eval1.log
	echo -e "quant_post_eval1,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/quant_post_eval1.log  ${log_path}/SUCCESS/quant_post_eval1.log
	echo -e "quant_post_eval1,SUCCESS" >>${result_path}/result.log
fi
# 量化后eval
time (python eval.py --model_path ./quant_model_train/MobileNet >${log_path}/quant_post_eval2.log) >>${log_path}/quant_post_eval2.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/quant_post_eval2.log ${log_path}/FAIL/quant_post_eval2.log
	echo -e "quant_post_eval2,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/quant_post_eval2.log  ${log_path}/SUCCESS/quant_post_eval2.log
	echo -e "quant_post_eval2,SUCCESS" >>${result_path}/result.log
fi
# 3 prune
cd ${current_dir}/demo/prune
wget http://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV1_pretrained.tar
tar -xf MobileNetV1_pretrained.tar
if [ -d "models" ];then
    rm -rf models
fi
CUDA_VISIBLE_DEVICES=3 python train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" --pretrained_model ./MobileNetV1_pretrained/ --num_epochs 1 >${current_dir}/prune_v1_T_1card 2>&1
cd ${current_dir}
cat prune_v1_T_1card |grep Final |awk -F ' ' 'END{print "kpis\tprune_v1_acc_top1_gpu1\t"$8"\nkpis\tprune_v1_acc_top5_gpu1\t"$10}' |tr -d ";" | python _ce.py

cd ${current_dir}/demo/prune
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" --pretrained_model ./MobileNetV1_pretrained/ --num_epochs 1 >${current_dir}/prune_v1_T_8card 2>&1
cd ${current_dir}
cat prune_v1_T_8card |grep Final |awk -F ' ' 'END{print "kpis\tprune_v1_acc_top1_gpu8\t"$8"\nkpis\tprune_v1_acc_top5_gpu8\t"$10}' |tr -d ";" | python _ce.py


#4 nas
# 4.1 sa_nas_mobilenetv2
cd ${current_dir}/demo/nas
time (CUDA_VISIBLE_DEVICES=3 python sa_nas_mobilenetv2.py --search_steps 1 --port 8881 >${log_path}/sa_nas_v2_T_1card.log) >>${log_path}/sa_nas_v2_T_1card.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/sa_nas_v2_T_1card.log ${log_path}/FAIL/sa_nas_v2_T_1card.log
	echo -e "sa_nas_v2_T_1card,train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/sa_nas_v2_T_1card.log  ${log_path}/SUCCESS/sa_nas_v2_T_1card.log
	echo -e "sa_nas_v2_T_1card,train,SUCCESS" >>${result_path}/result.log
fi

time (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python sa_nas_mobilenetv2.py --search_steps 1 --port 8882 >${log_path}/sa_nas_v2_T_8card.log) >>${log_path}/sa_nas_v2_T_8card.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/sa_nas_v2_T_8card.log ${log_path}/FAIL/sa_nas_v2_T_8card.log
	echo -e "sa_nas_v2_T_8card,train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/sa_nas_v2_T_8card.log  ${log_path}/SUCCESS/sa_nas_v2_T_8card.log
	echo -e "sa_nas_v2_T_8card,train,SUCCESS" >>${result_path}/result.log
fi

# 4.2 block_sa_nas_mobilenetv2
time (CUDA_VISIBLE_DEVICES=3 python block_sa_nas_mobilenetv2.py --search_steps 1 --port 8883 >${log_path}/block_sa_nas_v2_T_1card.log) >>${log_path}/block_sa_nas_v2_T_1card.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/block_sa_nas_v2_T_1card.log ${log_path}/FAIL/block_sa_nas_v2_T_1card.log
	echo -e "block_sa_nas_v2_T_1card,train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/block_sa_nas_v2_T_1card.log  ${log_path}/SUCCESS/block_sa_nas_v2_T_1card.log
	echo -e "block_sa_nas_v2_T_1card,train,SUCCESS" >>${result_path}/result.log
fi
time (CUDA_VISIBLE_DEVICES=0,1,2,3 python block_sa_nas_mobilenetv2.py --search_steps 1 --port 8884 >${log_path}/block_sa_nas_v2_T_8card.log) >>${log_path}/block_sa_nas_v2_T_8card.log 2>&1
#time (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python block_sa_nas_mobilenetv2.py --search_steps 1 --port 8883 >${log_path}/block_sa_nas_v2_T_8card.log) >>${log_path}/block_sa_nas_v2_T_8card.log 2>&1
if [ $? -ne 0 ];then
	mv ${log_path}/block_sa_nas_v2_T_8card.log ${log_path}/FAIL/block_sa_nas_v2_T_8card.log
	echo -e "block_sa_nas_v2_T_8card,train,FAIL" >>${result_path}/result.log;
else
	mv ${log_path}/block_sa_nas_v2_T_8card.log  ${log_path}/SUCCESS/block_sa_nas_v2_T_8card.log
	echo -e "block_sa_nas_v2_T_8card,train,SUCCESS" >>${result_path}/result.log
fi
