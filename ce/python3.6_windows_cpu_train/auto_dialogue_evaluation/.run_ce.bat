@echo off 
set FLAGS_sync_nccl_allreduce=0
set FLAGS_eager_delete_tensor_gb=1.0
set ce_mode=1 
python -u main.py --do_train=true --use_cuda=false --loss_type="CLS"  --max_seq_len=50  --save_model_path="data/saved_models/matching_pretrained"  --save_param="params"  --training_file="data/input/data/unlabel_data/train.ids" --epoch=20 --print_step=1 --save_step=400 --batch_size=256  --hidden_size=256 --emb_size=256  --vocab_size=484016 --learning_rate=0.001 --sample_pro=0.1 --enable_ce="store_true" | python _ce.py





