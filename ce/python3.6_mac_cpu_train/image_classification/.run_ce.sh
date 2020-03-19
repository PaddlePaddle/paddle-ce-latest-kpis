#!/bin/bash
# This file is only used for continuous evaluation.
model_list='AlexNet DPN107 DarkNet53 DenseNet121 EfficientNet HRNet_W18_C GoogLeNet InceptionV4 Xception65_deeplab MobileNetV1 MobileNetV2 ResNet50 ResNet152_vd Res2Net50_vd_26w_4s ResNeXt101_32x4d ResNeXt101_32x8d_wsl SE_ResNeXt50_vd_32x4d ShuffleNetV2_swish SqueezeNet1_1 VGG19'
for model in $model_list
do
python train.py --model=$model --num_epochs=1 --batch_size 8 --lr_strategy=cosine_decay --random_seed 1000 --use_gpu False --enable_ce=True > $model.log 2>&1
cat  $model.log | grep "train_cost_card1" | awk '{print "kpis\t""'$model'""_loss_card1\t"$5}' | python _ce.py
cat  $model.log | grep "train_speed_card1" | awk '{print "kpis\t""'$model'""_time_card1\t"$5}' | python _ce.py
# eval
python eval.py --model=$model --batch_size=32 --pretrained_model=output/$model/0 --use_gpu false >$log_path/$model_E.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/$model_E.log ${log_path}/FAIL/$model_E.log↩
        echo -e "$model,infer,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/$model_E.log ${log_path}/SUCCESS/$model_E.log↩
        echo -e "$model,infer,SUCCESS" >>${log_path}/result.log↩
fi
# infer
python infer.py --model=$model --pretrained_model=output/$model/0 --use_gpu False --data_dir=data/ILSVRC2012/test >$log_path/$model_I.log 2>&1
f [ $? -ne 0 ];then↩
        mv ${log_path}/$model_I.log ${log_path}/FAIL/$model_I.log↩
        echo -e "$model,infer,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/$model_I.log ${log_path}/SUCCESS/$model_I.log↩
        echo -e "$model,infer,SUCCESS" >>${log_path}/result.log↩
fi
# save_inference
python infer.py  --model=$model --use_gpu False --pretrained_model=output/$model/0 --save_inference=True >$log_path/$model_SI.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/$model_SI.log ${log_path}/FAIL/$model_SI.log↩
        echo -e "$model,infer,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/$model_SI.log ${log_path}/SUCCESS/$model_SI.log↩
        echo -e "$model,infer,SUCCESS" >>${log_path}/result.log↩
fi
# predict
python predict.py  --model_file=$model/model --params_file=$model/params  --image_path=data/ILSVRC2012/test/ILSVRC2012_val_00000001.jpeg --gpu_id=-1  --gpu_mem=1024 >$log_path/$model_P.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/$model_P.log ${log_path}/FAIL/$model_P.log↩
        echo -e "$model,infer,FAIL" >>${log_path}/result.log;↩
else↩
        mv ${log_path}/$model_P.log ${log_path}/SUCCESS/$model_P.log↩
        echo -e "$model,infer,SUCCESS" >>${log_path}/result.log↩
fi
done
