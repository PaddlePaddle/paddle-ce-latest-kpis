#!/bin/bash

bash run.sh
cat ptb_lm.log | python _ce.py
