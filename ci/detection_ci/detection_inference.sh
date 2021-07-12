#mkdir logs
if [ -d "logs" ];then rm -rf logs
fi
mkdir logs
if [ -d "logs_cpp" ];then rm -rf logs_cpp
fi
mkdir logs_cpp
#machine type
MACHINE_TYPE=`uname -m`
echo "MACHINE_TYPE: "${MACHINE_TYPE}
config_list='ppyolo_r50vd_dcn_1x_coco ppyolov2_r50vd_dcn_365e_coco yolov3_darknet53_270e_coco solov2_r50_fpn_1x_coco faster_rcnn_r50_fpn_1x_coco mask_rcnn_r50_1x_coco s2anet_conv_1x_dota ssd_mobilenet_v1_300_120e_voc ttfnet_darknet53_1x_coco fcos_r50_fpn_1x_coco'
config_s2anet='s2anet_conv_1x_dota'
mode_list='trt_fp32 trt_fp16 trt_int8 fluid'
for config in ${config_list}
do
image=demo/000000570688.jpg
if [[ -n `echo "${config_s2anet}" | grep -w "${config}"` ]];then
    image=demo/P0072__1.0__0___0.png
fi
model=`cat model_list | grep ${config}`
python tools/export_model.py \
       -c configs/${model} \
       --output_dir=inference_model \
       -o weights=https://paddledet.bj.bcebos.com/models/${config}.pdparams
for mode in ${mode_list}
do
if [[ ${mode} == 'trt_int8' ]];then
    trt_calib_mode=True
else
    trt_calib_mode=False
fi
python deploy/python/infer.py \
       --model_dir=./inference_model/${config} \
       --image_file=${image} \
       --device=GPU \
       --run_mode=${mode} \
       --threshold=0.5 \
       --trt_calib_mode=${trt_calib_mode} \
       --output_dir=python_infer_output/${config}_${mode} >logs/${config}_${mode}.log 2>&1
if [ $? -ne 0 ];then
    echo -e "${config}_${mode},python_infer,FAIL"
else
    echo -e "${config}_${mode},python_infer,SUCCESS"
fi
done
python deploy/python/infer.py \
       --model_dir=./inference_model/${config} \
       --image_file=${image} \
       --device=CPU \
       --threshold=0.5 \
       --output_dir=python_infer_output/${config}_cpu >logs/${config}_cpu.log 2>&1
if [ $? -ne 0 ];then
    echo -e "${config}_cpu,python_infer,FAIL"
else
    echo -e "${config}_cpu,python_infer,SUCCESS"
fi
python deploy/python/infer.py \
       --model_dir=./inference_model/${config} \
       --image_file=${image} \
       --device=CPU \
       --threshold=0.5 \
       --enable_mkldnn=True \
       --output_dir=python_infer_output/${config}_mkldnn >logs/${config}_mkldnn.log 2>&1
if [ $? -ne 0 ];then
    echo -e "${config}_mkldnn,python_infer,FAIL"
else
    echo -e "${config}_mkldnn,python_infer,SUCCESS"
fi
python deploy/python/infer.py \
       --model_dir=./inference_model/${config} \
       --image_dir=data \
       --device=GPU \
       --run_mode=fluid \
       --threshold=0.5 \
       --batch_size=2 \
       --output_dir=python_infer_output/${config}_batchsize_2 >logs/${config}_batchsize_2.log 2>&1
if [ $? -ne 0 ];then
    echo -e "${config}_bs2,python_infer,FAIL"
else
    echo -e "${config}_bs2,python_infer,SUCCESS"
fi
python deploy/python/infer.py \
       --model_dir=./inference_model/${config} \
       --video_file=video.mp4 \
       --device=GPU \
       --run_mode=fluid \
       --threshold=0.5 \
       --output_dir=python_infer_output/${config}_video >logs/${config}_video.log 2>&1
if [ $? -ne 0 ];then
    echo -e "${config}_video,python_infer,FAIL"
else
    echo -e "${config}_video,python_infer,SUCCESS"
fi
done

cd deploy/cpp
rm -rf paddle_inference
rm -rf deps/*
tar -xvf paddle_inference.tgz
mv paddle_inference_install_dir paddle_inference
sed -i "s|/path/to/paddle_inference|../paddle_inference|g" scripts/build.sh
sed -i "s|WITH_GPU=OFF|WITH_GPU=ON|g" scripts/build.sh
sed -i "s|WITH_TENSORRT=OFF|WITH_TENSORRT=ON|g" scripts/build.sh
sed -i "s|CUDA_LIB=/path/to/cuda/lib|CUDA_LIB=/usr/local/cuda/lib64|g" scripts/build.sh
if [[ "$MACHINE_TYPE" == "aarch64" ]]
then
sed -i "s|WITH_MKL=ON|WITH_MKL=OFF|g" scripts/build.sh
sed -i "s|TENSORRT_INC_DIR=/path/to/tensorrt/include|TENSORRT_INC_DIR=/usr/include/aarch64-linux-gnu|g" scripts/build.sh
sed -i "s|TENSORRT_LIB_DIR=/path/to/tensorrt/lib|TENSORRT_LIB_DIR=/usr/lib/aarch64-linux-gnu|g" scripts/build.sh
sed -i "s|CUDNN_LIB=/path/to/cudnn/lib|CUDNN_LIB=/usr/lib/aarch64-linux-gnu|g" scripts/build.sh
else
sed -i "s|TENSORRT_LIB_DIR=/path/to/tensorrt/lib|TENSORRT_LIB_DIR=/usr/local/TensorRT6-cuda10.1-cudnn7/lib|g" scripts/build.sh
sed -i "s|CUDNN_LIB=/path/to/cudnn/lib|CUDNN_LIB=/usr/lib/x86_64-linux-gnu|g" scripts/build.sh
fi
sh scripts/build.sh
cd ../..
for config in ${config_list}
do
image=demo/000000570688.jpg
if [[ -n `echo "${config_s2anet}" | grep -w "${config}"` ]];then
    image=demo/P0072__1.0__0___0.png
fi
for mode in ${mode_list}
do
if [[ ${mode} == 'trt_int8' ]];then
    trt_calib_mode=True
else
    trt_calib_mode=False
fi
./deploy/cpp/build/main --model_dir=inference_model/${config} --image_file=${image} --output_dir=cpp_infer_output/${config}_${mode} --device=GPU --run_mode=${mode} --threshold=0.5 --run_benchmark=True --trt_calib_mode=${trt_calib_mode} >logs_cpp/${config}_${mode}.log 2>&1
if [ $? -ne 0 ];then
    echo -e "${config}_${mode},cpp_infer,FAIL"
else
    echo -e "${config}_${mode},cpp_infer,SUCCESS"
fi
done
./deploy/cpp/build/main --model_dir=inference_model/${config} --image_file=${image} --output_dir=cpp_infer_output/${config}_cpu --device=CPU --threshold=0.5 --run_benchmark=True >logs_cpp/${config}_cpu.log 2>&1
if [ $? -ne 0 ];then
    echo -e "${config}_cpu,cpp_infer,FAIL"
else
    echo -e "${config}_cpu,cpp_infer,SUCCESS"
fi
./deploy/cpp/build/main --model_dir=inference_model/${config} --image_file=${image} --output_dir=cpp_infer_output/${config}_mkldnn --device=CPU --use_mkldnn=True --threshold=0.5 --run_benchmark=True >logs_cpp/${config}_mkldnn.log 2>&1
if [ $? -ne 0 ];then
    echo -e "${config}_mkldnn,cpp_infer,FAIL"
else
    echo -e "${config}_mkldnn,cpp_infer,SUCCESS"
fi
./deploy/cpp/build/main --model_dir=inference_model/${config} --image_dir=data --output_dir=cpp_infer_output/${config}_batchsize_2 --device=GPU --run_mode=fluid --batch_size=2 --threshold=0.5 --run_benchmark=True >logs_cpp/${config}_batchsize_2.log 2>&1
if [ $? -ne 0 ];then
    echo -e "${config}_bs2,cpp_infer,FAIL"
else
    echo -e "${config}_bs2,cpp_infer,SUCCESS"
fi
./deploy/cpp/build/main --model_dir=inference_model/${config} --video_file=video.mp4 --output_dir=cpp_infer_output/${config}_video --device=GPU --run_mode=fluid --threshold=0.5 >logs_cpp/${config}_video.log 2>&1
if [ $? -ne 0 ];then
    echo -e "${config}_video,cpp_infer,FAIL"
else
    echo -e "${config}_video,cpp_infer,SUCCESS"
fi
done
