@echo off
cd ../..

md log\text_matching_finetune
set logpath=%cd%\log\text_matching_finetune

cd models_repo\examples\text_matching\sentence_transformers\

python -m paddle.distributed.launch --gpus %2 train.py --device %1 --save_dir ./checkpoints  --epochs 1 > %logpath%/train_%1.log 2>&1
