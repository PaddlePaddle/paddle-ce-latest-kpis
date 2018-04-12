#!/usr/bin/env xonsh
import sys
# import argparse
# parser = argparse.ArgumentParser('train script')
# parser.add_argument('--train_cost_out', type=str)
# parser.add_argument('--valid_cost_out', type=str)
# parser.add_argument('--gpu', type=int, default=-1)
# args = parser.parse_args($ARGS[1:])

model_file = 'model.py'

python @(model_file) --batch_size 1000 --iters 10 --device CPU
