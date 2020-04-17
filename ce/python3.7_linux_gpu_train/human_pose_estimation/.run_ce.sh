export CUDA_VISIBLE_DEVICES=0
python train.py --dataset mpii --batch_size=16 --num_epochs=2  --enable_ce true --use_gpu true 1> log_card1
cat log_card1 | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --dataset mpii --num_epochs=2 --enable_ce true --use_gpu true 1> log_card8
cat log_card8 | python _ce.py

#eval
python val.py --dataset 'mpii' --checkpoint 'output/simplebase-mpii/0' --data_root 'data/mpii' >eval
if [ $? -ne 0 ];then 
	echo -e "human_pose,eval,FAIL"
else
	echo -e "human_pose,eval,SUCCESS"
fi 
