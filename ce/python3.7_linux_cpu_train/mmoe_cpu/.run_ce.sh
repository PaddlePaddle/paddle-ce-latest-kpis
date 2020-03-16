#!/bin/bash
# cpu
python mmoe_train.py >mmoe_cpu_T 2>&1
cat mmoe_cpu_T |awk -F ':| ' 'END{print "kpis\ttrain_loss_cpu\t"$4}' |tr -d '[|]|,' | python _ce.py
