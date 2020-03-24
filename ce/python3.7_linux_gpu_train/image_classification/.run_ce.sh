#!/bin/bash

# This file is only used for continuous evaluation.
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
#AlexNet 
train_AlexNet(){
python train.py \
       --enable_ce=True \
       --model=AlexNet \
       --batch_size=16 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --num_epochs=10 \
       --lr=0.01 \
       --l2_decay=1e-4 
}
#DPN
train_DPN107(){
python train.py \
       --enable_ce=True \
       --model=DPN107 \
       --batch_size=16 \
       --lr_strategy=cosine_decay \
       --lr=0.1 \
       --num_epochs=10 \
       --model_save_dir=output/ \
       --l2_decay=1e-4 \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1
}
#DarkNet
train_DarkNet53(){
python train.py \
       --enable_ce=True \
       --model=DarkNet53 \
       --batch_size=16 \
       --image_shape 3 256 256 \
       --lr_strategy=cosine_decay \
       --lr=0.1 \
       --num_epochs=10 \
       --model_save_dir=output/ \
       --l2_decay=1e-4 \
       --resize_short_size=256 \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1
}
#DenseNet121
train_DenseNet121(){
python train.py \
       --enable_ce=True \
       --model=DenseNet121 \
       --batch_size=16 \
       --lr_strategy=piecewise_decay \
       --lr=0.1 \
       --num_epochs=10 \
       --model_save_dir=output/ \
       --l2_decay=1e-4
}
#EfficientNet
train_EfficientNet(){
python train.py \
       --enable_ce=True \
       --model=EfficientNet \
       --batch_size=16 \
       --test_batch_size=128 \
       --resize_short_size=256 \
       --model_save_dir=output/ \
       --lr_strategy=exponential_decay_warmup \
       --lr=0.032 \
       --num_epochs=10 \
       --l2_decay=1e-5  
}
#GoogLeNet
train_GoogLeNet(){
python train.py \
       --enable_ce=True \
       --model=GoogLeNet \
       --batch_size=16 \
       --model_save_dir=output/ \
       --lr_strategy=cosine_decay \
       --lr=0.01 \
       --num_epochs=10 \
       --l2_decay=1e-4
}
#HRNet
train_HRNet_W18_C(){
python train.py \
       --enable_ce=True \
       --model=HRNet_W18_C \
       --batch_size=16 \
       --lr_strategy=piecewise_decay \
       --lr=0.1 \
       --num_epochs=10 \
       --model_save_dir=output/ \
       --l2_decay=1e-4
}
#InceptionV4
train_InceptionV4(){
python train.py \
       --enable_ce=True \
       --model=InceptionV4 \
       --batch_size=16 \
       --image_shape 3 299 299 \
       --lr_strategy=cosine_decay \
       --lr=0.045 \
       --num_epochs=10 \
       --model_save_dir=output/ \
       --l2_decay=1e-4 \
       --resize_short_size=320 \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1 
}
#MobileNetV1
train_MobileNetV1(){
python train.py \
       --enable_ce=True \
       --model=MobileNetV1 \
       --batch_size=16 \
       --model_save_dir=output/ \
       --lr_strategy=piecewise_decay \
       --num_epochs=10 \
       --lr=0.1 \
       --l2_decay=3e-5 
}
#MobileNetV2
train_MobileNetV2(){
python train.py \
        --enable_ce=True \
	--model=MobileNetV2 \
	--batch_size=16 \
	--model_save_dir=output/ \
	--lr_strategy=cosine_decay \
	--num_epochs=10 \
	--lr=0.1 \
	--l2_decay=4e-5
}
#Res2Net50_vd_26w_4s
train_Res2Net50_vd_26w_4s(){
python train.py \
       --enable_ce=True \
       --model=Res2Net50_vd_26w_4s \
       --batch_size=16 \
       --lr_strategy=cosine_decay \
       --lr=0.1 \
       --num_epochs=10 \
       --model_save_dir=output/ \
       --l2_decay=1e-4 \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1
}
#ResNeXt101
train_ResNeXt101_32x4d(){
python train.py \
        --enable_ce=True \
	--model=ResNeXt101_32x4d \
        --batch_size=16 \
        --lr_strategy=piecewise_decay \
        --lr=0.1 \
        --num_epochs=10 \
        --model_save_dir=output/ \
        --l2_decay=1e-4    
}
#ResNet
train_ResNet152_vd(){
python train.py \
       --enable_ce=True \
       --model=ResNet152_vd \
       --batch_size=16 \
       --lr_strategy=cosine_decay \
       --lr=0.1 \
       --num_epochs=10 \
       --model_save_dir=output/ \
       --l2_decay=1e-4 \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1
}
#SE_ResNeXt50
train_SE_ResNeXt50_vd_32x4d(){
python train.py \
       --enable_ce=True \
       --model=SE_ResNeXt50_vd_32x4d \
       --batch_size=16 \
       --lr_strategy=cosine_decay \
       --lr=0.1 \
       --num_epochs=10 \
       --model_save_dir=output/ \
       --l2_decay=1e-4 \
       --use_label_smoothing=True \
       --label_smoothing_epsilon=0.1 
}
#ShuffleNetV2
train_ShuffleNetV2_swish(){
python train.py \
        --enable_ce=True \
	--model=ShuffleNetV2_swish \
	--batch_size=16 \
	--model_save_dir=output/ \
	--lr_strategy=cosine_decay_warmup \
	--lr=0.5 \
	--num_epochs=10 \
	--l2_decay=4e-5
}
#SqueezeNet
train_SqueezeNet1_1(){
python train.py \
        --enable_ce=True \
        --model=SqueezeNet1_1 \
        --batch_size=16 \
        --lr_strategy=cosine_decay \
        --model_save_dir=output/ \
        --lr=0.02 \
        --num_epochs=10 \
        --l2_decay=1e-4
}
#VGG
train_VGG19(){
python train.py \
        --enable_ce=True \
	--model=VGG19 \
	--batch_size=16 \
	--lr_strategy=cosine_decay \
	--lr=0.01 \
	--num_epochs=10 \
        --model_save_dir=output/ \
	--l2_decay=4e-4
}
#Xception
train_Xception65_deeplab(){
python train.py \
       --enable_ce=True \
       --model=Xception65_deeplab \
       --batch_size=16 \
       --image_shape 3 299 299 \
       --lr_strategy=cosine_decay \
       --lr=0.045 \
       --num_epochs=10 \
       --model_save_dir=output/ \
       --l2_decay=1e-4 \
       --resize_short_size=320
}

model_list='AlexNet DPN107 DarkNet53 DenseNet121 EfficientNet GoogLeNet HRNet_W18_C InceptionV4 MobileNetV1 MobileNetV2 Res2Net50_vd_26w_4s ResNeXt101_32x4d ResNet152_vd SE_ResNeXt50_vd_32x4d ShuffleNetV2_swish SqueezeNet1_1 VGG19 Xception65_deeplab'

for model in ${model_list}
do

export CUDA_VISIBLE_DEVICES=7
train_${model} > log_${model}_card1 2>&1
cat log_${model}_card1 | grep "train_cost_card1" | tail -1 | awk '{print "kpis\t""'$model'""_loss_card1\t"$5}' | python _ce.py
cat log_${model}_card1 | grep "train_speed_card1" | tail -1 | awk '{print "kpis\t""'$model'""_time_card1\t"$5}' | python _ce.py

#eval
python eval.py \
       --model=${model} \
       --pretrained_model=output/${model}/0 \
       --data_dir=./data/ILSVRC2012/ \
       --batch_size=32 >eval_${model}
if [ $? -ne 0 ];then
	echo -e "${model},eval,FAIL"
else
	echo -e "${model},eval,SUCCESS"
fi

#infer
python infer.py \
       --model=${model} \
       --pretrained_model=output/${model}/0 \
       --class_map_path=./utils/tools/readable_label.txt \
       --image_path=data/ILSVRC2012/val/ILSVRC2012_val_00050000.JPEG >infer_${model}
if [ $? -ne 0 ];then
        echo -e "${model},infer,FAIL"
else
        echo -e "${model},infer,SUCCESS"
fi

#export
python infer.py \
       --model=${model} \
       --pretrained_model=output/${model}/0 \
       --save_inference=True >export_${model}
if [ $? -ne 0 ];then
        echo -e "${model},export,FAIL"
else
        echo -e "${model},export,SUCCESS"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
train_${model} > log_${model}_card8 2>&1
cat log_${model}_card8 | grep "train_cost_card8" | tail -1 | awk '{print "kpis\t""'$model'""_loss_card8\t"$5}' | python _ce.py
cat log_${model}_card8 | grep "train_speed_card8" | tail -1 | awk '{print "kpis\t""'$model'""_time_card8\t"$5}' | python _ce.py

done
