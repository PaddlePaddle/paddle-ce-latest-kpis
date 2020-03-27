#!/bin/bash
data_dir='dataset/ILSVRC2012/'
is_full_quantize=False
model_save_dir='outputs/'
pretrain_dir='./pretrain'
batch_size=16
num_epochs=1
major_quant_ops="conv2d,mul"
minor_quant_ops="elementwise_add"

train(){
python quant.py \
       --model=${model}\
       --pretrained_fp32_mode=./pretrain/${model}_pretrained \
       --num_epochs=${num_epochs} \
       --batch_size=${batch_size} \
       --model_save_dir=./outputs \
       --use_gpu=True \
       --data_dir=${data_dir} \
       --total_images=1281167 \
       --class_dim=1000 \
       --image_shape=3,224,224 \
       --lr_strategy=piecewise_decay \
       --lr=0.0001 \
       --act_quant_type=moving_average_abs_max \
       --wt_quant_type=abs_max \
       --is_full_quantize=${is_full_quantize} \
       --weight_bits=8 \
       --activation_bits=8 \
       --has_weight_quant_op_type="conv2d,depthwise_conv2d,mul,matmul" \
       --no_weight_quant_op_type="elementwise_add,pool2d"
}

export CUDA_VISIBLE_DEVICES=0,1,2,3

for model_name in GoogleNet MobileNet MobileNetV2 ResNet50 VGG16
do
    model=${model_name}
    train &> ${model_name}_log.txt
    cat ${model_name}_log.txt | python _ce.py
done