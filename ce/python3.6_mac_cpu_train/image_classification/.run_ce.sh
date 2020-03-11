#!/bin/bash
# This file is only used for continuous evaluation.
model_list='AlexNet DPN107 DarkNet53 DenseNet121 EfficientNet HRNet_W18_C GoogLeNet InceptionV4 Xception65_deeplab MobileNetV1 MobileNetV2 ResNet50 ResNet152_vd Res2Net50_vd_26w_4s ResNeXt101_32x4d ResNeXt101_32x8d_wsl SE_ResNeXt50_vd_32x4d ShuffleNetV2_swish SqueezeNet1_1 VGG19'
for model in $model_list
do
python train.py --model=$model --num_epochs=1 --batch_size 8 --lr_strategy=cosine_decay --random_seed 1000 --use_gpu False --enable_ce=True > $model.log 2>&1
cat  $model.log | grep "train_cost_card1" | awk '{print "kpis\t""'$model'""_loss_card1\t"$5}' | python _ce.py
cat  $model.log | grep "train_speed_card1" | awk '{print "kpis\t""'$model'""_time_card1\t"$5}' | python _ce.py
done
