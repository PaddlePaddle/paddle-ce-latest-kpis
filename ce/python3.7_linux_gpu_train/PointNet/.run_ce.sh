#! /bin/bash
cls_train() {
    python train_cls.py \
        --model=${model_type} \
        --batch_size=32 \
        --save_dir=checkpoints_${model_type}_cls \
        --epoch 1 \
        --enable_ce
}
seg_train() {
    python train_seg.py \
        --model=${model_type} \
        --batch_size=32 \
        --save_dir=checkpoints_${model_type}_seg \
        --epoch 1 \
        --enable_ce
}

ce_run() {
    cls_log=log_cls_${model_type}_${#cudaid[@]}card
    cls_train 1> ${cls_log}
    cat ${cls_log} | python _ce.py

    seg_log=log_seg_${model_type}_${#cudaid[@]}card
    seg_train 1> ${seg_log}
    cat ${seg_log} | python _ce.py
}

eval(){
    python eval_seg.py \
        --model=${model_type} \
        --weights=checkpoints_${model_type}_seg/0 >eval_${model_type}_seg
    if [ $? -ne 0 ];then
        echo -e "${model_type}_seg,eval,FAIL"
    else
        echo -e "${model_type}_seg,eval,SUCCESS"
    fi
    python eval_cls.py \
        --model=${model_type} \
        --weights=checkpoints_${model_type}_cls/0 >eval_${model_type}_cls
    if [ $? -ne 0 ];then
        echo -e "${model_type}_cls,eval,FAIL"
    else
        echo -e "${model_type}_cls,eval,SUCCESS"
    fi
}

cudaid=${pointnet_1:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

model_type=MSG
ce_run
eval
model_type=SSG
ce_run
eval
cudaid=${pointnet_4:=0,1,2,3} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

model_type=MSG
ce_run

model_type=SSG
ce_run

