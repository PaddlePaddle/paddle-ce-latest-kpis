export CUDA_VISIBLE_DEVICES=0
python train.py --dataset mpii --batch_size=16 --num_epochs=2  --enable_ce true --use_gpu true 1> log_card1
cat log_card1 | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --dataset mpii --num_epochs=2 --enable_ce true --use_gpu true 1> log_card8
cat log_card8 | python _ce.py 
