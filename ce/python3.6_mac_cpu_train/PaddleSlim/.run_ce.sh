#!/bin/bash
# This file is only used for continuous evaluation.

export FLAGS_eager_delete_tensor_gb=0.0
sed -i s/"    epoch: 200"/"    epoch: 1"/g configs/filter_pruning_uniform.yaml
python compress.py --use_gpu False --model "MobileNet" --pretrained_model ./pretrain/MobileNetV1_pretrained --batch_size=64 --compress_config ./configs/filter_pruning_uniform.yaml 2>&1 | tee run.log | python _ce.py
	
	







