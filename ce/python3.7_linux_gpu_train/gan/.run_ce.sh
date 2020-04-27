#!/bin/bash

# This file is only used for continuous evaluation.
train_CGAN(){
python train.py \
       --model_net CGAN \
       --dataset mnist \
       --noise_size 100 \
       --batch_size 121 \
       --epoch 1 \
       --output ./output/cgan/ 
}
train_DCGAN(){
python train.py \
       --model_net DCGAN \
       --dataset mnist \
       --noise_size 100 \
       --batch_size 128 \
       --epoch 1 \
       --output ./output/dcgan/
}
train_CycleGAN(){
python train.py \
       --model_net CycleGAN \
       --dataset cityscapes \
       --batch_size 1 \
       --net_G resnet_9block \
       --g_base_dim 32 \
       --net_D basic \
       --norm_type batch_norm \
       --epoch 1 \
       --image_size 286 \
       --crop_size 256 \
       --crop_type Random \
       --output ./output/cyclegan/
}
train_SPADE(){
python train.py \
       --model_net SPADE \
       --dataset cityscapes_spade \
       --train_list ./data/cityscapes_spade/train_list \
       --test_list ./data/cityscapes_spade/val_list \
       --crop_type Random \
       --batch_size 1 \
       --epoch 1 \
       --load_height 612 \
       --load_width 1124 \
       --crop_height 512 \
       --crop_width 1024 \
       --label_nc 36 \
       --output ./output/spade/           
}
train_AttGAN(){
python train.py \
       --model_net AttGAN \
       --dataset celeba \
       --crop_size 170 \
       --image_size 128 \
       --train_list ./data/celeba/list_attr_celeba.txt \
       --gan_mode wgan \
       --batch_size 32 \
       --epoch 1 \
       --dis_norm instance_norm \
       --output ./output/attgan/
}
train_StarGAN(){
python train.py \
       --model_net StarGAN \
       --dataset celeba \
       --crop_size 178 \
       --image_size 128 \
       --train_list ./data/celeba/list_attr_celeba.txt \
       --batch_size 32 \
       --epoch 1 \
       --gan_mode wgan \
       --output ./output/stargan/
}
train_STGAN(){
python train.py \
       --model_net STGAN \
       --dataset celeba \
       --crop_size 170 \
       --image_size 128 \
       --train_list ./data/celeba/list_attr_celeba.txt \
       --gan_mode wgan \
       --batch_size 32 \
       --epoch 1 \
       --dis_norm instance_norm \
       --output ./output/stgan/
}
train_Pix2pix(){
python train.py \
       --model_net Pix2pix \
       --dataset cityscapes \
       --train_list data/cityscapes/pix2pix_train_list \
       --test_list data/cityscapes/pix2pix_test_list \
       --crop_type Random \
       --dropout True \
       --gan_mode vanilla \
       --batch_size 1 \
       --epoch 1 \
       --image_size 286 \
       --crop_size 256 \
       --output ./output/pix2pix/
}
infer_CGAN(){
python infer.py \
       --model_net CGAN \
       --init_model ./output/cgan/checkpoints/0/ \
       --n_samples 32 \
       --noise_size 100 \
       --output ./infer_result/cgan/      
}
infer_DCGAN(){
python infer.py \
       --model_net DCGAN \
       --init_model ./output/dcgan/checkpoints/0/ \
       --n_samples 32 \
       --noise_size 100 \
       --output ./infer_result/dcgan/      
}
infer_CycleGAN(){
python infer.py \
       --model_net CycleGAN \
       --init_model ./output/cyclegan/checkpoints/0/ \
       --dataset_dir data/cityscapes/ \
       --image_size 256 \
       --n_samples 1 \
       --crop_size 256 \
       --input_style B \
       --test_list ./data/cityscapes/testB.txt \
       --net_G resnet_9block \
       --g_base_dims 32 \
       --output ./infer_result/cyclegan/
}
infer_SPADE(){
python infer.py \
       --model_net SPADE \
       --test_list ./data/cityscapes_spade/test_list \
       --load_height 512 \
       --load_width 1024 \
       --crop_height 512 \
       --crop_width 1024 \
       --dataset_dir ./data/cityscapes_spade/ \
       --init_model ./output/spade/checkpoints/0/ \
       --output ./infer_result/spade/
}
infer_AttGAN(){
python infer.py \
       --model_net AttGAN \
       --init_model ./output/attgan/checkpoints/0/ \
       --dataset_dir ./data/celeba/ \
       --image_size 128 \
       --output ./infer_result/attgan/
}
infer_StarGAN(){
python infer.py \
       --model_net StarGAN \
       --init_model ./output/stargan/checkpoints/0/ \
       --dataset_dir ./data/celeba/ \
       --image_size 128 \
       --c_dim 5 \
       --selected_attrs "Black_Hair,Blond_Hair,Brown_Hair,Male,Young" \
       --output ./infer_result/stargan/
}
infer_STGAN(){
python infer.py \
       --model_net STGAN \
       --init_model ./output/stgan/checkpoints/0/ \
       --dataset_dir ./data/celeba/ \
       --image_size 128 \
       --use_gru True \
       --output ./infer_result/stgan/
}
infer_Pix2pix(){
python infer.py \
       --model_net Pix2pix \
       --init_model ./output/pix2pix/checkpoints/0/ \
       --image_size 256 \
       --n_samples 1 \
       --crop_size 256 \
       --dataset_dir data/cityscapes/ \
       --net_G unet_256 \
       --test_list data/cityscapes/testB.txt \
       --output ./infer_result/pix2pix/

}
#train

model_list='CGAN DCGAN CycleGAN SPADE AttGAN StarGAN STGAN Pix2pix'
for model in ${model_list}
do
export CUDA_VISIBLE_DEVICES=7
train_${model} > log_${model} 2>&1
if [[ ${model} == CGAN || ${model} == DCGAN ]];then
cat log_${model} | grep "loss" | tail -1 | awk '{print "kpis\t""'$model'""_dloss\t"$8"\nkpis\t""'$model'""_gloss\t"$10"\nkpis\t""'$model'""_time\t"$12}' | python _ce.py
elif [[ ${model} == AttGAN || ${model} == StarGAN || ${model} == STGAN ]];then
cat log_${model} | grep "d_loss" | tail -1 | tr ";" " " | awk '{print "kpis\t""'$model'""_dloss\t"$8}' | python _ce.py
cat log_${model} | grep "g_loss" | tail -1 | tr ";" " " | awk '{print "kpis\t""'$model'""_gloss\t"$6}' | python _ce.py
cat log_${model} | grep "Batch_time_cost" | tail -1 | tr ";" " " | awk '{print "kpis\t""'$model'""_time\t"$2}' | python _ce.py
elif [[ ${model} == Pix2pix || ${model} == SPADE ]];then
cat log_${model} | grep "d_loss" | tail -1 | tr ";" " " | awk '{print "kpis\t""'$model'""_dloss\t"$2}' | python _ce.py
cat log_${model} | grep "g_loss" | tail -1 | tr ";" " " | awk '{print "kpis\t""'$model'""_gloss\t"$2}' | python _ce.py
cat log_${model} | grep "Batch_time_cost" | tail -1 | tr ";" " " | awk '{print "kpis\t""'$model'""_time\t"$2}' | python _ce.py
else
cat log_${model} | grep "d_A_loss" | tail -1 | tr ";" " " | awk '{print "kpis\t""'$model'""_dloss\t"$2"\nkpis\t""'$model'""_gloss\t"$4"\nkpis\t""'$model'""_time\t"$12}' | python _ce.py
cat log_${model} | grep "Batch_time_cost" | tail -1 | tr ";" " " | awk '{print "kpis\t""'$model'""_time\t"$2}' | python _ce.py
fi
done

#infer
for model in ${model_list}
do
export CUDA_VISIBLE_DEVICES=7
infer_${model} >infer_${model} 2>&1
if [ $? -ne 0 ];then
    echo -e "${model},infer,FAIL"
else
    echo -e "${model},infer,SUCCESS"
fi
done
