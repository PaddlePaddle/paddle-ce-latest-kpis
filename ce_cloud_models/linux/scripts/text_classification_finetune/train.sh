#unset http_proxy
HTTPPROXY=$http_proxy
HTTPSPROXY=$https_proxy
unset http_proxy
unset https_proxy

#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型训练阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/text_classification/pretrained_models/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/models_repo

#访问RD程序
cd $code_path

DEVICE=$1
if [[ ${DEVICE} == "gpu" ]]; then
N_GPU=1
else
N_GPU=0
fi
MULTI=$2
if [[ ${MULTI} == "multi" ]]; then
N_GPU=2
fi

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
fi
}

python -m paddle.distributed.launch --gpus $3 train.py \
  --device ${DEVICE} \
  --save_dir ./checkpoints \
  --epochs 1 >$log_path/train_$2_${DEVICE}.log 2>&1
print_info $? train_${DEVICE}


#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
