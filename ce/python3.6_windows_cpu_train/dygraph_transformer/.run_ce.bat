@echo off
set CUDA_VISIBLE_DEVICES=0
rem transformer
python -u train.py --epoch 1 --src_vocab_fpath data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 --trg_vocab_fpath data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 --training_file data/wmt16_ende_data_bpe_clean/newstest2016.tok.bpe.32000.en-de  --batch_size 1024 --print_step 1 --use_cuda false --random_seed 102 --enable_ce true | python _ce.py






