@echo off
cd ../..

md log\stacl
set logpath=%cd%\log\stacl

cd models_repo\examples\simultaneous_translation\stacl\

python -m paddle.distributed.launch --gpus %2 train.py --config ./config/transformer.yaml > %logpath%/train_%1.log 2>&1