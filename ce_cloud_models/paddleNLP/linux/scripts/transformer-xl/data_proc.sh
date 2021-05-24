
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"

#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../models_repo/examples/language_model/$model_name

#获取数据逻辑
mkdir -p $code_path/data/
#wget -P $code_path http://10.21.226.155:8687/openwebtext.tar.xz

#数据处理逻辑
cd $code_path

#取消代理
#HTTPPROXY=$http_proxy
#HTTPSPROXY=$https_proxy
#unset http_proxy
#unset https_proxy

if [ ! -d "gen_data" ]
then
    sed -i "s/python3 prep_enwik8.py/python3.7 prep_enwik8.py/g"  ./gen_data.sh
    bash gen_data.sh
fi

#export http_proxy=$HTTPPROXY
#export https_proxy=$HTTPSPROXY



