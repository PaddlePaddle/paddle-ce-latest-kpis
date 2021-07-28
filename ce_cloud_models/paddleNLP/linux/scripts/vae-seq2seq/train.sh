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
code_path=$cur_path/../../models_repo/examples/text_generation/vae-seq2seq/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改
cd $root_path/models_repo

#访问RD程序
cd $code_path

DEVICE=$1

MULTI=$2
if [[ ${MULTI} == "multi" ]]; then
    python -m paddle.distributed.launch train.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset ptb \
        --model_path ptb_model \
        --device ${DEVICE} \
        --max_epoch 1 >$log_path/train_${MULTI}_${DEVICE}.log 2>&1
else # 单卡或cpu
    python train.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset ptb \
        --model_path ptb_model\
        --device ${DEVICE} \
        --max_epoch 1 >$log_path/train_${MULTI}_${DEVICE}.log 2>&1
fi

#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
