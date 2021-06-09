#!/bin/bash
cur_path=`pwd`
#路径配置
root_path=$(dirname $(pwd))
img_path=$root_path/img_data/ocr_web2.png
hub run chinese_ocr_db_crnn_mobile --input_path $img_path
