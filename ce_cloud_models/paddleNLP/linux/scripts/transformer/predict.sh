#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型预测阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/machine_translation/$model_name
log_path=$root_path/log/$model_name/
if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi


#访问RD程序
cd $code_path

#使用动态图预测
python predict.py --config ./configs/transformer.base.yaml > $log_path/origin_predict.log
ls output_file

cat $log_path/origin_predict.log
#导出静态图预测模型与预测引擎预测 (后续考虑)
python export_model.py --config ./configs/transformer.base.yaml > $log_path/predict1.log
ls infer_model/
cat $log_path/predict1.log
#完成动转静的静态图的模型, 实现高性能预测的功能
python deploy/python/inference.py  --config ./configs/transformer.base.yaml > $log_path/predict2.log
cat $log_path/predict2.log
