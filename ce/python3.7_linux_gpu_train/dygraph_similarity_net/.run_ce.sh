#!/usr/bin/env bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
TASK_NAME='simnet'
TRAIN_DATA_PATH=./dataset/train_pairwise_data
VALID_DATA_PATH=./dataset/test_pairwise_data
TEST_DATA_PATH=./dataset/test_pairwise_data
INFER_DATA_PATH=./dataset/infer_data
VOCAB_PATH=./dataset/term2id.dict
CKPT_PATH=./model_files
TEST_RESULT_PATH=./test_result
INFER_RESULT_PATH=./infer_result
TASK_MODE='pairwise'
CONFIG_PATH=./config/cnn_pairwise.json

INIT_CHECKPOINT=./model_files/simnet_cnn_pairwise_pretrained_model/cnn_pairwise


# run_train
train() {
    python run_classifier.py \
		--task_name ${TASK_NAME} \
		--use_cuda True \
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
		--epoch 20 \
		--save_steps 2000 \
		--validation_steps 1 \
		--compute_accuracy False \
		--lamda 0.958 \
		--task_mode ${TASK_MODE}\
		--init_checkpoint "" \
                --enable_ce
}


export CUDA_VISIBLE_DEVICES=0
train | python _ce.py
sleep 20
export CUDA_VISIBLE_DEVICES=0,1,2,3
train | python _ce.py
