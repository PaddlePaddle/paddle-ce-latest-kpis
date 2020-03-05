@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set CUDA_VISIBLE_DEVICES=0
rem train
python -u -u main.py --do_train True --epoch 1 --src_vocab_fpath data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 --trg_vocab_fpath data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 --training_file data/wmt16_ende_data_bpe_clean/newstest2016.tok.bpe.32000.en-de  --batch_size 1024  --n_head 16  --d_model 1024 --d_inner_hid 4096  --prepostprocess_dropout 0.3 --print_step 1 --enable_ce true | python _ce.py


