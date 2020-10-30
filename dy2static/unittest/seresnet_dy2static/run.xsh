#!/bin/bash

bash train.sh
cat seresnet.log | python _ce.py
