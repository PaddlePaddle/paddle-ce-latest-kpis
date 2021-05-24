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

echo "$model_name 模型样例测试阶段"

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

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}
python predict.py \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset ptb \
        --device ${DEVICE} \
        --infer_output_file infer_output.txt \
        --init_from_ckpt ptb_model/final > $log_path/infer_${DEVICE}.log 2>&1

print_info $? infer_${DEVICE}


#set http_proxy
export http_proxy=$HTTPPROXY
export https_proxy=$HTTPSPROXY
