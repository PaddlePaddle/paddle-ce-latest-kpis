@echo off
cd ../..

set logpath=%cd%\log\text_matching_finetune

cd models_repo\examples\text_matching\sentence_transformers\

python predict.py --device %1 --params_path checkpoints\model_5000\model_state.pdparams > %logpath%/infer_%1.log 2>&1