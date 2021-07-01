cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型评估阶段"
#路径配置
root_path=$cur_path/../../
log_path=$root_path/log/$model_name/

if [ ! -d $log_path ]; then
  mkdir -p $log_path
fi

print_info(){
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

#访问RD程序
cd $root_path/models_repo/examples/simultaneous_translation/stacl

sed -r 's/(@@ )|(@@ ?$)//g' predict.txt > predict.tok.txt

git clone https://github.com/moses-smt/mosesdecoder.git

perl mosesdecoder/scripts/generic/multi-bleu.perl newstest2017.tok.en < predict.tok.txt > $log_path/eval_$2_$1.log 2>&1

print_info $? eval_$2_$1
