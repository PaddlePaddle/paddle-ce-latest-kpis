#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleSpeech/DeepVoice3/. ./
if [ ! -d "data" ];then
ln -s ${data_path}/DeepVoice3 data
fi
pip install -r requirements.txt
python -c "import nltk;nltk.download('punkt');nltk.download('cmudict')"
./.run_ce.sh
