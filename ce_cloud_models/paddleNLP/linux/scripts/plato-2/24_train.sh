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
code_path=$cur_path/../../models_repo/examples/dialogue/plato-2/
log_path=$root_path/log/$model_name/
mkdir -p $log_path

#访问RD程序
cd $code_path

wget https://paddlenlp.bj.bcebos.com/models/transformers/plato2/24L.pdparams

if [[ $1 == "gpu" ]]; then
     python interaction.py\
     --vocab_path ./data/vocab.txt\
     --spm_model_file ./data/spm.model\
     --num_layers 24\
     --init_from_ckpt ./24L.pdparams > $log_path/train_24_$2_$1.log 2>&1
fi