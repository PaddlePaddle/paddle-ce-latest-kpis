#!/bin/bash

cd utils
bash download_thirdparty.sh
cd ..
cd src
cp -r * ../
cp __init__.py ../../
./.run_ce.sh
cp *_factor.txt ../
