#!/bin/bash

bash train.sh
cat mnist.log | python _ce.py
