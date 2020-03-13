@echo off
rem ptb_lm
python ptb_dy.py --data_path data/simple-examples/data --ce --model_type small --use_gpu False | python _ce.py
              







