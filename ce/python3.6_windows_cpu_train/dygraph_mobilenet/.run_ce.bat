@echo off
set FLAGS_cudnn_deterministic=True
set CUDA_VISIBLE_DEVICES=0
rem mobilentV1
python train.py  --batch_size=10 --total_images=1281167  --class_dim=1000 --image_shape=3,224,224 --model_save_dir=output.v1.sing/ --lr_strategy=piecewise_decay --lr=0.1   --data_dir=./data/ILSVRC2012  --l2_decay=3e-5  --model=MobileNetV1	--num_epochs=1 --use_gpu false	> mobilenetv1.log
type mobilenetv1.log | grep "epoch 0"| gawk "{print \"kpis\tMobileNetV1_train_loss_card1\t\"$8}" | python _ce.py
type mobilenetv1.log | grep "final eval"| gawk "{print \"kpis\tMobileNetV1_eval_loss_card1\t\"$4}" | python _ce.py
rem mobilentV2
python train.py  --batch_size=10 --total_images=1281167 --class_dim=1000 --image_shape=3,224,224 --model_save_dir=output.v2.sing/ --lr_strategy=cosine_decay --lr=0.1  --num_epochs=240  --data_dir=./data/ILSVRC2012 --l2_decay=4e-5  --model=MobileNetV2 --num_epochs=1 --use_gpu false > mobilenetv2.log
type mobilenetv2.log | grep "epoch 0"| gawk "{print \"kpis\tMobileNetV2_train_loss_card1\t\"$8}" | python _ce.py
type mobilenetv2.log | grep "final eval"| gawk "{print \"kpis\tMobileNetV2_eval_loss_card1\t\"$4}" | python _ce.py



