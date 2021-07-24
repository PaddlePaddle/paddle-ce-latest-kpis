@echo off
cd ../..

set logpath=%cd%\log\few_shot_ptuing

cd models_repo\examples\few_shot\p-tuning\

md  output\%3

python -u -m paddle.distributed.launch --gpus %2 predict.py --task_name %3 --device %1 --init_from_ckpt "./checkpoints/%3/model_%4/model_state.pdparams" --output_dir "./output/%3" --batch_size 16 --max_seq_length 512 > %logpath%/infer_%3_%1.log 2>&1