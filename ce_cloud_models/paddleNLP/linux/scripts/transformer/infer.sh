cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型样例测试阶段"
#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/machine_translation/$model_name
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

#访问RD程序
cd $code_path
python generate_sample.py > $log_path/infer.log 2>&1
cat $log_path/infer.log
