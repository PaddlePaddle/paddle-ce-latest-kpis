#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型评估阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/machine_translation/$model_name
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi
#访问RD程序
cd $code_path
python predict.py --config ./configs/transformer.base.yaml > $log_path/predict.log 2>&1