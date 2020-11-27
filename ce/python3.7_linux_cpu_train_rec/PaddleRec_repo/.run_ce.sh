#!/bin/bash
current_dir=$PWD
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid1=${card2:=0,1} # use 0-th card as default
pip install pandas
pip install jieba


mkdir ${log_path}/rec
export log_path_rec=${log_path}/rec
print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path_rec}/$2 ${log_path_rec}/$2_FAIL.log
    echo -e "\033[31m $2_FAIL \033[0m"
else
    mv ${log_path_rec}/$2 ${log_path_rec}/$2_SUCCESS.log
    echo -e "\033[32m $2_SUCCESS \033[0m"
fi
}
python setup.py install
# 1 contentunderstanding (3/3)
run_con_cpu(){
cp ${dataset_path}/rec_repo/rec_config/$1_cpu_config.yaml ./
python -m paddlerec.run -m ./$1_cpu_config.yaml >${log_path_rec}/$1_cpu 2>&1
print_info $? $1_cpu
}
run_con_gpu(){
cp ${dataset_path}/rec_repo/rec_config/$1_gpu_config.yaml ./
CUDA_VISIBLE_DEVICES=${cudaid1} python -m paddlerec.run -m ./$1_gpu_config.yaml >${log_path_rec}/$1_gpu1 2>&1
print_info $? $1_gpu1
}

contentunderstanding_cpu(){
# 1.1 tagspace
model=tagspace
cd ${current_dir}/models/contentunderstanding/tagspace
mv data data_bk
ln -s ${dataset_path}/rec_repo/tagspace/data
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -1|awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$7}' |python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_acc_cpu\t"$20"\nkpis\t""'${model}'""_loss_cpu\t"$23}'|tr -d '[][]' |python _ce.py

# 1.2 textcnn
model=textcnn
cd ${current_dir}/models/contentunderstanding/textcnn
ln -s ${dataset_path}/rec_repo/textcnn/senta_data
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -1|awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$7}' |python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_acc_cpu\t"$20"\nkpis\t""'${model}'""_loss_cpu\t"$23}'|tr -d '[][]' |python _ce.py

# 1.3 textcnn_pretrain
model=textcnn_pretrain
cd ${current_dir}/models/contentunderstanding/textcnn_pretrain
ln -s ${dataset_path}/rec_repo/textcnn/senta_data
ln -s ${dataset_path}/rec_repo/textcnn/pretrain/pretrain_model
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -1|awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$7}' |python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_acc_cpu\t"$20"\nkpis\t""'${model}'""_loss_cpu\t"$23}'|tr -d '[][]' |python _ce.py
}

match_cpu(){
# 2 match(3/3)
# 2.1 match-pyramid  这个目录下的模型都是eval中写死了训练的log路径，需要改进
model=match-pyramid
model_tmp=match_pyramid
cd ${current_dir}/models/match/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_cpu ${model}
python eval.py >${log_path_rec}/${model}_cpu_eval 2>&1
print_info $? ${model}_cpu_eval
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -2|tail -1 |awk -F ' ' '{print "kpis\t""'${model_tmp}'""_epoch_time_cpu\t"$6}'|python _ce.py
cat ${log_path_rec}/${model}_cpu_eval_SUCCESS.log |grep map|awk -F ' ' '{print "kpis\t""'${model_tmp}'""_map_cpu\t"$2}'|python _ce.py

# 2.2 dssm
model=dssm
cd ${current_dir}/models/match/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
cp run.sh run_cpu.sh && cp run.sh run_gpu.sh
sed -i 's/config/dssm_cpu_config/g' run_cpu.sh
bash run_cpu.sh > ${log_path_rec}/${model}_cpu 2>&1
print_info $? ${model}_cpu
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |tail -2 |head -1|awk -F ' ' '{print "kpis\t""'${model}'""_pos_neg_cpu\t"$2}'|python _ce.py

# 2.3 multiview-simnet
model=multiview-simnet
model_tmp=multiview_simnet
cd ${current_dir}/models/match/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
cp run.sh run_cpu.sh && cp run.sh run_gpu.sh
sed -i 's/config/multiview-simnet_cpu_config/g' run_cpu.sh
bash run_cpu.sh > ${log_path_rec}/${model}_cpu 2>&1
print_info $? ${model}_cpu
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |tail -2 |head -1|awk -F ' ' '{print "kpis\t""'${model_tmp}'""_pos_neg_cpu\t"$2}'|python _ce.py
}

multitask_cpu(){
# 3.1 multitask (2/4)  esmm
model=esmm
cd ${current_dir}/models/multitask/esmm
mv data data_bk
ln -s ${dataset_path}/rec_repo/esmm/data
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -9|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$7}'|python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_AUC_ctr_cpu\t"$15"\nkpis\t""'${model}'""_AUC_ctcvr_cpu\t"$18}'|tr -d '[]'|python _ce.py

# 3.2 mmoe
model=mmoe
cd ${current_dir}/models/multitask/mmoe
mv data data_bk
ln -s ${dataset_path}/rec_repo/mmoe/data
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_AUC_income_cpu\t"$15"\nkpis\t""'${model}'""_AUC_marital_cpu\t"$18}'|tr -d '[][]' |python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$7}' |python _ce.py
}

rank_cpu(){
# 4.1 rank(5/21) deepfm
model=deepfm
cd ${current_dir}/models/rank/deepfm
mv data data_bk
ln -s ${dataset_path}/rec_repo/deepfm/data
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_cpu\t"$5}' |tr -d '[]'|python _ce.py

# 4.2 dnn
model=dnn
cd ${current_dir}/models/rank/dnn
mv data data_bk
ln -s ${dataset_path}/rec_repo/dnn/data
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_cpu\t"$5}' |tr -d '[]'|python _ce.py

# 4.3 fm
model=fm
cd ${current_dir}/models/rank/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_cpu\t"$5}' |tr -d '[]'|python _ce.py

# 4.4 lr
model=logistic_regression
cd ${current_dir}/models/rank/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_cpu\t"$5}' |tr -d '[]'|python _ce.py

# 4.5 wide_deep
model=wide_deep
cd ${current_dir}/models/rank/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_cpu\t"$5"\nkpis\t""'${model}'""_acc_cpu\t"$7}' |tr -d '[]'|python _ce.py
}

recall_cpu(){
# 5 recall (3/8)
# 5.1 gnn
model=gnn
cd ${current_dir}/models/recall/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_cpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |awk -F ' |=' 'END{print "kpis\t""'${model}'""_recall20_cpu\t"$17}' |tr -d '[]'|python _ce.py

# 5.2 word2vec
model=word2vec
cd ${current_dir}/models/recall/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_cpu ${model}
python infer.py --test_dir ./data/all_test --dict_path ./data/all_dict/word_id_dict.txt \
--batch_size 10000 --model_dir ./increment_w2v_cpu/  \
--start_index 0 --last_index 4 --emb_size 300 >${log_path_rec}/${model}_infer_cpu 2>&1
print_info $? ${model}_infer_cpu
cd ${current_dir}
cat ${log_path_rec}/${model}_cpu_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_infer_cpu_SUCCESS.log |grep acc|awk -F ' |:' 'END{print "kpis\t""'${model}'""_acc_cpu\t"$5}' |tr -d '[]'|python _ce.py

# 5.3 youtube_dnn
model=youtube_dnn
cd ${current_dir}/models/recall/${model}
sh data_prepare.sh
run_con_cpu ${model}
print_info $? ${model}
python infer.py --test_epoch 19 --inference_model_dir ./inference_youtubednn_cpu \
--increment_model_dir ./increment_youtubednn_cpu --watch_vec_size 64 \
--search_vec_size 64 --other_feat_size 64 --topk 5 >${log_path_rec}/${model}_infer_cpu 2>&1
print_info $? ${model}_infer_cpu
}


################################################################################################
contentunderstanding_gpu1(){
# 1.1 tagspace  1epoch
model=tagspace
cd ${current_dir}/models/contentunderstanding/tagspace
mv data data_bk
ln -s ${dataset_path}/rec_repo/tagspace/data
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -1|awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$7}' |python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_acc_gpu1\t"$20"\nkpis\t""'${model}'""_loss_gpu1\t"$23}'|tr -d '[][]' |python _ce.py

# 1.2 textcnn
model=textcnn
cd ${current_dir}/models/contentunderstanding/textcnn
ln -s ${dataset_path}/rec_repo/textcnn/senta_data
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -1|awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$7}' |python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_acc_gpu1\t"$20"\nkpis\t""'${model}'""_loss_gpu1\t"$23}'|tr -d '[][]' |python _ce.py

# 1.3 textcnn_pretrain
model=textcnn_pretrain
cd ${current_dir}/models/contentunderstanding/textcnn_pretrain
ln -s ${dataset_path}/rec_repo/textcnn/senta_data
ln -s ${dataset_path}/rec_repo/textcnn/pretrain/pretrain_model
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -1|awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$7}' |python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_acc_gpu1\t"$20"\nkpis\t""'${model}'""_loss_gpu1\t"$23}'|tr -d '[][]' |python _ce.py
}

match_gpu1(){
# 2 match(3/3)
# 2.1 match-pyramid  这个目录下的模型都是eval中写死了训练的log路径，需要改进
model=match-pyramid
model_tmp=match_pyramid
cd ${current_dir}/models/match/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_gpu ${model}
python eval.py >${log_path_rec}/${model}_gpu1_eval 2>&1
print_info $? ${model}_gpu1_eval
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -2|tail -1 |awk -F ' ' '{print "kpis\t""'${model_tmp}'""_epoch_time_gpu1\t"$6}'|python _ce.py
cat ${log_path_rec}/${model}_gpu1_eval_SUCCESS.log |grep map|awk -F ' ' '{print "kpis\t""'${model_tmp}'""_map_gpu1\t"$2}'|python _ce.py

# 2.2 dssm
model=dssm
cd ${current_dir}/models/match/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
cp run.sh run_cpu.sh && cp run.sh run_gpu.sh
sed -i 's/config/dssm_gpu1_config/g' run_gpu.sh
CUDA_VISIBLE_DEVICES=${cudaid1} bash run_gpu.sh > ${log_path_rec}/${model}_gpu1 2>&1
print_info $? ${model}_gpu1
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |tail -2 |head -1|awk -F ' ' '{print "kpis\t""'${model}'""_pos_neg_gpu1\t"$2}'|python _ce.py

# 2.3 multiview-simnet
model=multiview-simnet
model_tmp=multiview_simnet
cd ${current_dir}/models/match/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
cp run.sh run_cpu.sh && cp run.sh run_gpu.sh
sed -i 's/config/multiview-simnet_gpu1_config/g' run_gpu.sh

CUDA_VISIBLE_DEVICES=${cudaid1} bash run_gpu.sh > ${log_path_rec}/${model}_gpu1 2>&1
print_info $? ${model}_gpu1
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |tail -2 |head -1|awk -F ' ' '{print "kpis\t""'${model_tmp}'""_pos_neg_gpu1\t"$2}'|python _ce.py
}

multitask_gpu1(){
# 3.1 multitask (2/4)  esmm
model=esmm
cd ${current_dir}/models/multitask/esmm
mv data data_bk
ln -s ${dataset_path}/rec_repo/esmm/data
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -9|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$7}'|python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_AUC_ctr_gpu1\t"$15"\nkpis\t""'${model}'""_AUC_ctcvr_gpu1\t"$18}'|tr -d '[]'|python _ce.py


# 3.2 mmoe
model=mmoe
cd ${current_dir}/models/multitask/mmoe
mv data data_bk
ln -s ${dataset_path}/rec_repo/mmoe/data
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_AUC_income_gpu1\t"$15"\nkpis\t""'${model}'""_AUC_marital_gpu1\t"$18}'|tr -d '[][]' |python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$7}' |python _ce.py
}

rank_gpu1(){
# 4.1 rank(5/21) deepfm
model=deepfm
cd ${current_dir}/models/rank/deepfm
mv data data_bk
ln -s ${dataset_path}/rec_repo/deepfm/data
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_gpu1\t"$5}' |tr -d '[]'|python _ce.py

# 4.2 dnn
model=dnn
cd ${current_dir}/models/rank/dnn
mv data data_bk
ln -s ${dataset_path}/rec_repo/dnn/data
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_gpu1\t"$5}' |tr -d '[]'|python _ce.py

# 4.3 fm
model=fm
cd ${current_dir}/models/rank/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_gpu1\t"$5}' |tr -d '[]'|python _ce.py


# 4.4 lr
model=logistic_regression
cd ${current_dir}/models/rank/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_gpu1\t"$5}' |tr -d '[]'|python _ce.py


# 4.5 wide_deep
model=wide_deep
cd ${current_dir}/models/rank/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_gpu1\t"$5"\nkpis\t""'${model}'""_acc_gpu1\t"$7}' |tr -d '[]'|python _ce.py
}

recall_gpu1(){
# 5 recall (3/8)
# 5.1 gnn
model=gnn
cd ${current_dir}/models/recall/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |awk -F ' |=' 'END{print "kpis\t""'${model}'""_recall20_gpu1\t"$17}' |tr -d '[]'|python _ce.py

# 5.2 word2vec
model=word2vec
cd ${current_dir}/models/recall/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_gpu ${model}
python infer.py --test_dir ./data/all_test --dict_path ./data/all_dict/word_id_dict.txt \
--batch_size 10000 --model_dir ./increment_w2v_gpu/  \
--start_index 0 --last_index 4 --emb_size 300 >${log_path_rec}/${model}_infer_gpu1 2>&1
print_info $? ${model}_infer_gpu1
cd ${current_dir}
cat ${log_path_rec}/${model}_gpu1_SUCCESS.log |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path_rec}/${model}_infer_gpu1_SUCCESS.log |grep acc|awk -F ' |:' 'END{print "kpis\t""'${model}'""_acc_gpu1\t"$5}' |tr -d '[]'|python _ce.py

# 5.3 youtube_dnn
model=youtube_dnn
cd ${current_dir}/models/recall/${model}
sh data_prepare.sh
run_con_gpu ${model}
print_info $? ${model}
CUDA_VISIBLE_DEVICES=${cudaid1} python infer.py --use_gpu 1 --test_epoch 19 \
--inference_model_dir ./inference_youtubednn_gpu --increment_model_dir ./increment_youtubednn_gpu \
--watch_vec_size 64 --search_vec_size 64 \
--other_feat_size 64 --topk 5 >${log_path_rec}/${model}_infer_gpu1 2>&1
print_info $? ${model}_infer_gpu1
}

contentunderstanding_cpu
match_cpu
multitask_cpu
rank_cpu
recall_cpu

#contentunderstanding_gpu1
#match_gpu1
#multitask_gpu1
#rank_gpu1
#recall_gpu1
