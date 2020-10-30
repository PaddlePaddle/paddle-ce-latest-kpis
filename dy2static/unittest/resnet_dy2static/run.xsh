#!/bin/bash

bash train.sh
cat resnet.log | python _ce.py
