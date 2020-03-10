#!/bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.5
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1

python train.py --data-root=data/ljspeech --preexport="preexports/deepvoice3_ljspeech.json" --hparams="checkpoint_interval=815,eval_interval=815,nepochs=1" | python _ce.py

