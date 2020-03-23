#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${face_detection:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --batch_size=2 --epoc_num=1 --batch_num=200 --parallel=False --enable_ce >log_1card 
cat log_1card | python _ce.py


cudaid=${face_detection_m:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python train.py --batch_size=8 --epoc_num=1 --batch_num=200 --parallel=False --enable_ce >log_4cards
cat log_4cards | python _ce.py

#eval
python -u widerface_eval.py --model_dir=output/0 --pred_dir=pred >eval
if [ $? -ne 0 ];then
	echo -e "face_detection,eval,FAIL"
else
	echo -e "face_detection,eval,SUCCESS"
fi
#infer
python widerface_eval.py --infer=True --confs_threshold=0.15 --model_dir=output/0/ --image_path=data/WIDER_train/images/1--Handshaking/1_Handshaking_Handshaking_1_102.jpg >infer
if [ $? -ne 0 ];then
        echo -e "face_detection,infer,FAIL"
else
        echo -e "face_detection,infer,SUCCESS"
fi
