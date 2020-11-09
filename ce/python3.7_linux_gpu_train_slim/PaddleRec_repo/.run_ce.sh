#!/bin/bash
current_dir=$PWD
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid1=${card1:=2} # use 0-th card as default

print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2 ${log_path}/$2_FAIL.log
    echo -e "\033[31m ${log_path}/$2_FAIL \033[0m"
else
    mv ${log_path}/$2 ${log_path}/$2_SUCCESS.log
    echo -e "\033[32m ${log_path}/$2_SUCCESS \033[0m"
fi
}
python setup.py install
# 1 contentunderstanding
run_con_cpu(){
cp ${dataset_path}/rec_repo/rec_config/$1_cpu_config.yaml ./
python -m paddlerec.run -m ./$1_cpu_config.yaml >${log_path}/$1_cpu 2>&1
#print_info $? $1
}
run_con_gpu(){
cp ${dataset_path}/rec_repo/rec_config/$1_gpu_config.yaml ./
CUDA_VISIBLE_DEVICES=$cudaid1 python -m paddlerec.run -m ./$1_gpu_config.yaml >${log_path}/$1_gpu1 2>&1
#print_info $? $1
}
# 1.1 tagspace
model=tagspace
cd ${current_dir}/models/contentunderstanding/tagspace
mv data data_bk
ln -s ${dataset_path}/rec_repo/tagspace/data
run_con_cpu ${model}
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path}/${model}_cpu |grep done |head -10|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$7"\nkpis\t""'${model}'""_acc_cpu\t"$12"\nkpis\t""'${model}'""_loss_cpu\t"$15}'|tr -d '[][]' |python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |head -10|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$7"\nkpis\t""'${model}'""_acc_gpu1\t"$12"\nkpis\t""'${model}'""_loss_gpu1\t"$15}'|tr -d '[][]' |python _ce.py

# 1.2 textcnn
model=textcnn
cd ${current_dir}/models/contentunderstanding/textcnn
ln -s ${dataset_path}/rec_repo/textcnn/senta_data
run_con_cpu ${model}
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path}/${model}_cpu |grep done |head -16|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$7"\nkpis\t""'${model}'""_acc_cpu\t"$12"\t""'${model}'""_loss_cpu\t"$15}'|tr -d '[][]' |python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |head -16|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$7"\nkpis\t""'${model}'""_acc_gpu1\t"$12"\t""'${model}'""_loss_gpu1\t"$15}'|tr -d '[][]' |python _ce.py


# 1.3 textcnn_pretrain
model=textcnn_pretrain
cd ${current_dir}/models/contentunderstanding/textcnn_pretrain
ln -s ${dataset_path}/rec_repo/textcnn/senta_data
ln -s ${dataset_path}/rec_repo/textcnn/pretrain/pretrain_model
run_con_cpu ${model}
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path}/${model}_cpu |grep done |head -6|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$7"\nkpis\t""'${model}'""_acc_cpu\t"$12"\nkpis\t""'${model}'""_loss_cpu\t"$15}'|tr -d '[][]' |python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |head -6|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$7"\nkpis\t""'${model}'""_acc_gpu1\t"$12"\nkpis\t""'${model}'""_loss_gpu1\t"$15}'|tr -d '[][]' |python _ce.py

# 3.1 multitask (1/3)  esmm
model=esmm
cd ${current_dir}/models/multitask/esmm
mv data data_bk
ln -s ${dataset_path}/rec_repo/esmm/data
run_con_cpu ${model}
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path}/${model}_cpu |grep done |head -6|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$7"\nkpis\t"'${model}'"_acc_cpu\t"$12"\nkpis\t""'${model}'""_loss_cpu\t"$15}'|tr -d '[][]' |python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |head -6|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$7"\nkpis\t""'${model}'""_acc_gpu1\t"$12"\nkpis\t""'${model}'""_loss_gpu1\t"$15}'|tr -d '[][]' |python _ce.py

# 3.2 mmoe
model=mmoe
cd ${current_dir}/models/multitask/mmoe
mv data data_bk
ln -s ${dataset_path}/rec_repo/mmoe/data
run_con_cpu ${model}
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path}/${model}_cpu |grep done |head -6|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$7"\nkpis\t""'${model}'""_acc_cpu\t"$12"\nkpis\t""'${model}'""_loss_cpu\t"$15}'|tr -d '[][]' |python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |head -6|tail -1 |awk -F ' |,|=' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$7"\nkpis\t""'${model}'""_acc_gpu1\t"$12"\nkpis\t""'${model}'""_loss_gpu1\t"$15}'|tr -d '[][]' |python _ce.py

# 4.1 rank(5) deepfm
model=deepfm
cd ${current_dir}/models/rank/deepfm
mv data data_bk
ln -s ${dataset_path}/rec_repo/deepfm/data
run_con_cpu ${model}
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path}/${model}_cpu |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path}/${model}_cpu |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_cpu\t"$4}' |tr -d '[]'|python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_gpu1\t"$4}' |tr -d '[]'|python _ce.py

# 4.2 dnn
model=dnn
cd ${current_dir}/models/rank/dnn
mv data data_bk
ln -s ${dataset_path}/rec_repo/dnn/data
run_con_cpu ${model}
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path}/${model}_cpu |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path}/${model}_cpu |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_cpu\t"$4}' |tr -d '[]'|python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_gpu1\t"$4}' |tr -d '[]'|python _ce.py

# 4.3 fm
model=fm
cd ${current_dir}/models/rank/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_cpu ${model}
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path}/${model}_cpu |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path}/${model}_cpu |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_cpu\t"$4}' |tr -d '[]'|python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_gpu1"\t$4}' |tr -d '[]'|python _ce.py


# 4.4 lr
model=logistic_regression
cd ${current_dir}/models/rank/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_cpu ${model}
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path}/${model}_cpu |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path}/${model}_cpu |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_cpu\t"$4}' |tr -d '[]'|python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_gpu1\t"$4}' |tr -d '[]'|python _ce.py


# 4.5 wide_deep
model=wide_deep
cd ${current_dir}/models/rank/${model}
mv data data_bk
ln -s ${dataset_path}/rec_repo/${model}/data
run_con_cpu ${model}
run_con_gpu ${model}
cd ${current_dir}
cat ${log_path}/${model}_cpu |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_cpu\t"$6}' |tr -d ','|python _ce.py
cat ${log_path}/${model}_cpu |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_cpu\t"$4}' |tr -d '[]'|python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |head -1 |awk -F ' ' '{print "kpis\t""'${model}'""_epoch_time_gpu1\t"$6}' |tr -d ','|python _ce.py
cat ${log_path}/${model}_gpu1 |grep done |awk -F ',|=' 'END{print "kpis\t""'${model}'""_auc_gpu1\t"$4}' |tr -d '[]'|python _ce.py


