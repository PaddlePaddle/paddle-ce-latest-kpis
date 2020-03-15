export ce_mode=1
rm -f *_factor.txt
python train.py --batch_size=32 --total_step=100 --eval_period=100 --log_period=100 --save_model_period=100 --use_gpu=True 1> ./tmp.log
cat tmp.log | python _ce.py
rm tmp.log
#eval
wget https://paddle-ocr-models.bj.bcebos.com/ocr_ctc.zip
unzip ocr_ctc.zip
python eval.py --model_path="./ocr_ctc/ocr_ctc_params" >eval
if [ $? -ne 0 ];then
    echo -e "ocr,eval,FAIL"
else
    echo -e "ocr,eval,SUCCESS"
fi
#infer
python infer.py --model_path="ocr_ctc/ocr_ctc_params" >infer
if [ $? -ne 0 ];then
    echo -e "ocr,infer,FAIL"
else
    echo -e "ocr,infer,SUCCESS"
fi
