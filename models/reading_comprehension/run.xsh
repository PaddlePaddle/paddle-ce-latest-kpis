#!/bin/bash

cd utils
bash download_thirdparty.sh
cd ..
cd src
cp -r ../latest_kpis ./
./.run_ce.sh
