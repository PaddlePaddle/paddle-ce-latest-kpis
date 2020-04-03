#!/usr/bin/env bash
TASK_NAME='simnet'
TRAIN_DATA_PATH=./data/train_pointwise_data
VALID_DATA_PATH=./data/test_pointwise_data
TEST_DATA_PATH=./data/test_pointwise_data
INFER_DATA_PATH=./data/infer_data
VOCAB_PATH=./data/term2id.dict
CKPT_PATH=./model_files
TEST_RESULT_PATH=./test_result
INFER_RESULT_PATH=./infer_result
TASK_MODE='pointwise'
CONFIG_PATH=./config/bow_pointwise.json
INIT_CHECKPOINT=./model_files/bow_pointwise/1000


# run_train
train() {
	python run_classifier.py \
		--task_name ${TASK_NAME} \
		--use_cuda False \
		--do_train True \
		--do_valid True \
		--do_test True \
		--do_infer True \
		--batch_size 128 \
		--train_data_dir ${TRAIN_DATA_PATH} \
		--valid_data_dir ${VALID_DATA_PATH} \
		--test_data_dir ${TEST_DATA_PATH} \
		--infer_data_dir ${INFER_DATA_PATH} \
		--output_dir ${CKPT_PATH} \
		--config_path ${CONFIG_PATH} \
		--vocab_path ${VOCAB_PATH} \
		--epoch 1 \
		--save_steps 1000 \
		--validation_steps 100 \
		--compute_accuracy False \
		--lamda 0.958 \
		--task_mode ${TASK_MODE} \
                --enable_ce \
                --init_checkpoint ""
}

export CUDA_VISIBLE_DEVICES=0
export CPU_NUM=10
train 1> log 
cat log | python _ce.py
