#!/usr/bin/env xonsh
import sys

model_file = 'mnist.py'

python @(model_file) --batch_size 128 --pass_num 5 --device CPU
