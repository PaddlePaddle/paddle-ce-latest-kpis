#!/bin/bash

bash train.sh
cat reinforcement_learning.log | python _ce.py
