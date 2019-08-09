#!/bin/bash
model='ResNet50'
pretrained_model_dir='/models/pretrained_model_dir/ResNet50_pretrained'
inference_model_root='/models/pretrained_model_dir/inference_model/'
data_path=''
use_gpu=false
class_dim=1000
image_shape='3,224,224'
batch_size=1
 
python .infer_ce.py --pretrained_model_dir=${pretrained_model_dir} \
                     --model=${model} \
                     --inference_model_root=${inference_model_root} \
                     --data_path=${data_path} \
                     --use_gpu=${use_gpu} \
                     --class_dim=${class_dim} \
                     --image_shape=${image_shape} \
                     --batch_size=${batch_size}
 
nosetests -s -v --with-xunit --xunit-file=check_save_load_info.xml check_data.py
rm ./results_save_model.txt ./results_load_model.txt