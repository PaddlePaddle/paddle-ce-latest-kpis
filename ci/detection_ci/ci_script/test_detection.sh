!/bin/bash
#create log dir
if [ -d "log" ];then rm -rf log
fi
mkdir log
if [ -d "log_err" ];then rm -rf log_err
fi
mkdir log_err
#compile cpp_infer
cd deploy/cpp
# dynamic c++ compile
cp -r ../../paddle_inference.tgz .
tar xvf paddle_inference.tgz
sed -i "s|WITH_GPU=OFF|WITH_GPU=ON|g" scripts/build.sh
sed -i "s|WITH_TENSORRT=OFF|WITH_TENSORRT=ON|g" scripts/build.sh
sed -i "s|CUDA_LIB=/path/to/cuda/lib|CUDA_LIB=/usr/local/cuda/lib64|g" scripts/build.sh
sed -i "s|/path/to/paddle_inference|../paddle_inference_install_dir|g" scripts/build.sh
sed -i "s|TENSORRT_LIB_DIR=/path/to/tensorrt/lib|TENSORRT_LIB_DIR=/usr/local/TensorRT6-cuda10.1-cudnn7/lib|g" scripts/build.sh
sed -i "s|CUDNN_LIB=/path/to/cudnn/lib|CUDNN_LIB=/usr/lib/x86_64-linux-gnu|g" scripts/build.sh
sh scripts/build.sh
cd ../..
#compile op
cd ppdet/ext_op
python setup.py install 
cd ../..
# prepare dynamic data
sed -i "s/trainval.txt/test.txt/g" configs/datasets/voc.yml
# modify dynamic_train_iter
sed -i '/for step_id, data in enumerate(self.loader):/i\            max_step_id =1' ppdet/engine/trainer.py
sed -i '/for step_id, data in enumerate(self.loader):/a\                if step_id == max_step_id: break' ppdet/engine/trainer.py
#modify eval iter
sed -i '/for step_id, data in enumerate(loader):/i\        max_step_id =1' ppdet/engine/trainer.py
sed -i '/for step_id, data in enumerate(loader):/a\            if step_id == max_step_id: break' ppdet/engine/trainer.py
if [ -d 'dataset/coco' ];then
rm -rf dataset/coco
fi
ln -s ${file_path}/data/coco dataset/coco
if [ -d 'dataset/voc' ];then
rm -rf dataset/voc
fi
ln -s ${file_path}/data/pascalvoc dataset/voc
if [ -d "dataset/mot" ];then rm -rf dataset/mot
fi
ln -s ${file_path}/data/mot dataset/mot
if [ -d "dataset/DOTA_1024_s2anet" ];then rm -rf dataset/DOTA_1024_s2anet
fi
ln -s ${file_path}/data/DOTA_1024_s2anet dataset/DOTA_1024_s2anet
print_result(){
    if [ $? -ne 0 ];then
        echo -e "${model},${model_type},${mode},FAIL"
        cd log_err
        if [ ! -d ${model} ];then
            mkdir ${model}
        fi
        cd ../${model_path}
        mv log/${model}/${model}_${model_type}_${mode}.log log_err/${model}/
        err_sign=true
    else
        echo -e "${model},${model_type},${mode},SUCCESS"
    fi
}
TRAIN(){
    export CUDA_VISIBLE_DEVICES=$cudaid2
    mode=train
    python -m paddle.distributed.launch \
    tools/train.py \
           -c configs/${config} \
           -o TrainReader.batch_size=1 epoch=1 >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
TRAIN_CPU(){
    mode=train_cpu
    python -m paddle.distributed.launch \
    tools/train.py \
           -c configs/${config} \
           -o TrainReader.batch_size=1 epoch=1 use_gpu=false >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
TRAIN_WITH_EVAL(){
    mode=train_with_eval
    python -m paddle.distributed.launch \
    tools/train.py \
           -c configs/${config} \
           -o TrainReader.batch_size=1 epoch=1 --eval >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
EVAL(){
    export CUDA_VISIBLE_DEVICES=$cudaid1
    mode=eval
    python tools/eval.py \
           -c configs/${config} \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams EvalReader.batch_size=1 >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
EVAL_bs2(){
    export CUDA_VISIBLE_DEVICES=$cudaid1
    mode=eval_bs2
    python tools/eval.py \
           -c configs/${config} \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams EvalReader.batch_size=2 >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
EVAL_MOT(){
    export CUDA_VISIBLE_DEVICES=$cudaid1
    mode=eval
    python tools/eval_mot.py \
           -c configs/${config} \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
EVAL_MOT_bs2(){
    export CUDA_VISIBLE_DEVICES=$cudaid1
    mode=eval_bs2
    python tools/eval_mot.py \
           -c configs/${config} \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
INFER(){
    export CUDA_VISIBLE_DEVICES=$cudaid1
    mode=infer
    python tools/infer.py \
           -c configs/${config} \
           --infer_img=${image} \
           --output_dir=infer_output/${model}/ \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
INFER_MOT(){
    export CUDA_VISIBLE_DEVICES=$cudaid1
    mode=infer
    python tools/infer_mot.py \
           -c configs/${config} \
           --video_file=video.mp4 \
           --output_dir=infer_output/${model}/ \
           --save_videos \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
EXPORT(){
    mode=export
    python tools/export_model.py \
           -c configs/${config} \
           --output_dir=inference_model \
           -o weights=https://paddledet.bj.bcebos.com/models/${weight_dir}${model}.pdparams >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
PYTHON_INFER(){
    mode=python_infer
    export CUDA_VISIBLE_DEVICES=$cudaid1
    python deploy/python/infer.py \
           --model_dir=inference_model/${model} \
           --image_file=${image} \
           --run_mode=fluid \
           --device=GPU \
           --threshold=0.5 \
           --output_dir=python_infer_output/${model}/ >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
CPP_INFER(){
    mode=cpp_infer
    ./deploy/cpp/build/main \
        --model_dir=inference_model/${model} \
        --image_file=${image} \
        --device=GPU \
        --run_mode=fluid \
        --threshold=0.5 \
        --output_dir=cpp_infer_output/${model} >log/${model}/${model}_${model_type}_${mode}.log 2>&1
    print_result
}
model_list='ppyolov2_r50vd_dcn_365e_coco yolov3_darknet53_270e_coco solov2_r50_fpn_1x_coco faster_rcnn_r50_fpn_1x_coco mask_rcnn_r50_1x_coco cascade_rcnn_r50_fpn_1x_coco s2anet_conv_1x_dota ssd_mobilenet_v1_300_120e_voc ttfnet_darknet53_1x_coco fcos_r50_fpn_1x_coco hrnet_w32_256x192 fairmot_dla34_30e_1088x608'
model_s2anet='s2anet_conv_1x_dota'
model_mot='fairmot_dla34_30e_1088x608'
model_keypoint='hrnet_w32_256x192'
model_ppyolov2='ppyolov2_r50vd_dcn_365e_coco'
err_sign=false
model_type=dynamic
for model in ${model_list}
do
weight_dir=
if [[ -n `echo "${model_keypoint}" | grep -w "${model}"` ]];then
    weight_dir=keypoint/
elif
   [[ -n `echo "${model_mot}" | grep -w "${model}"` ]];then
    weight_dir=mot/
fi
image=demo/000000570688.jpg
if [[ -n `echo "${model_s2anet}" | grep -w "${model}"` ]];then
    image=demo/P0072__1.0__0___0.png
elif [[ -n `echo "${model_keypoint}" | grep -w "${model}"` ]];then
    image=demo/hrnet_demo.jpg
fi
config=`cat model_list_ci | grep ${model}`
cd log && mkdir ${model} && cd ..
TRAIN
TRAIN_CPU
if [[ -n `echo "${model_ppyolov2}" | grep -w "${model}"` ]];then
    echo -e "skip train with eval for model ${model}!"
else
    TRAIN_WITH_EVAL
fi
if [[ -n `echo "${model_mot}" | grep -w "${model}"` ]];then
    EVAL_MOT
    EVAL_MOT_bs2
    INFER_MOT
else
    EVAL
    EVAL_bs2
    INFER
fi
EXPORT
if [[ -n `echo "${model_mot}" | grep -w "${model}"` ]] || [[ -n `echo "${model_keypoint}" | grep -w "${model}"` ]];then
    echo -e "The model ${model} not support inference deployment!"    
else
    PYTHON_INFER
    CPP_INFER
fi
done
if [ "${err_sign}" = true ];then
    exit 1
fi
