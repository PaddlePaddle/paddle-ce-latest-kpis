@echo off
set CUDA_VISIBLE_DEVICES=0
rem det_mv3_db
python tools/train.py -c configs/det/det_mv3_db.yml -o Global.use_gpu=False Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=60 > %log_path%/det_mv3_db_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\det_mv3_db_T.log  %log_path%\FAIL\det_mv3_db_T.log
        echo   det_mv3_db,train,FAIL  >> %log_path%\result.log
        echo   train of det_mv3_db failed!
) else (
        move  %log_path%\det_mv3_db_T.log  %log_path%\SUCCESS\det_mv3_db_T.log
        echo   det_mv3_db,train,SUCCISS  >> %log_path%\result.log
        echo   train of det_mv3_db successfully!
)
python tools/eval.py -c configs/det/det_mv3_db.yml  -o Global.use_gpu=False Global.checkpoints="output/det_db/best_accuracy" > %log_path%/det_mv3_db_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\det_mv3_db_E.log  %log_path%\FAIL\det_mv3_db_E.log
        echo   det_mv3_db,eval,FAIL  >> %log_path%\result.log
        echo   eval of det_mv3_db failed!
) else (
        move  %log_path%\det_mv3_db_E.log  %log_path%\SUCCESS\det_mv3_db_E.log
        echo   det_mv3_db,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of det_mv3_db successfully!
)
python tools/infer_det.py -c configs/det/det_mv3_db.yml -o Global.use_gpu=False TestReader.infer_img="./doc/imgs_en/" Global.checkpoints="output/det_db/best_accuracy" > %log_path%/det_mv3_db_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\det_mv3_db_I.log  %log_path%\FAIL\det_mv3_db_I.log
        echo   det_mv3_db,infer,FAIL  >> %log_path%\result.log
        echo   infer of det_mv3_db failed!
) else (
        move  %log_path%\det_mv3_db_I.log  %log_path%\SUCCESS\det_mv3_db_I.log
        echo   det_mv3_db,infer,SUCCISS  >> %log_path%\result.log
        echo   infer of det_mv3_db successfully!
)
python tools/export_model.py -c configs/det/det_mv3_db.yml -o Global.use_gpu=False Global.checkpoints="output/det_db/best_accuracy" Global.save_inference_dir="./inference_det/det_db" > %log_path%/det_mv3_db_export.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\det_mv3_db_export.log  %log_path%\FAIL\det_mv3_db_export.log
        echo   det_mv3_db,export model,FAIL  >> %log_path%\result.log
        echo   export model of det_mv3_db failed!
) else (
        move  %log_path%\det_mv3_db_export.log  %log_path%\SUCCESS\det_mv3_db_export.log
        echo   det_mv3_db,export model,SUCCISS  >> %log_path%\result.log
        echo   export model of det_mv3_db successfully!
)
python tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference_det/det_db" --det_algorithm=DB --use_gpu=False --use_tensorrt=False > %log_path%/det_mv3_db_P.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\det_mv3_db_P.log  %log_path%\FAIL\det_mv3_db_P.log
        echo   det_mv3_db,predict,FAIL  >> %log_path%\result.log
        echo   predict of det_mv3_db failed!
) else (
        move  %log_path%\det_mv3_db_P.log  %log_path%\SUCCESS\det_mv3_db_P.log
        echo   det_mv3_db,predict,SUCCISS  >> %log_path%\result.log
        echo   predict of det_mv3_db successfully!
)

rem det_r50_vd_east
python tools/train.py -c configs/det/det_r50_vd_east.yml -o Global.use_gpu=False Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=60 > %log_path%/det_r50_vd_east_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\det_r50_vd_east_T.log  %log_path%\FAIL\det_r50_vd_east_T.log
        echo   det_r50_vd_east,train,FAIL  >> %log_path%\result.log
        echo   train of det_r50_vd_east failed!
) else (
        move  %log_path%\det_r50_vd_east_T.log  %log_path%\SUCCESS\det_r50_vd_east_T.log
        echo   det_r50_vd_east,train,SUCCISS  >> %log_path%\result.log
        echo   train of det_r50_vd_east successfully!
)
python tools/eval.py -c configs/det/det_r50_vd_east.yml  -o Global.use_gpu=False Global.checkpoints="output/det_east/best_accuracy" > %log_path%/det_r50_vd_east_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\det_r50_vd_east_E.log  %log_path%\FAIL\det_r50_vd_east_E.log
        echo   det_r50_vd_east,eval,FAIL  >> %log_path%\result.log
        echo   eval of det_r50_vd_east failed!
) else (
        move  %log_path%\det_r50_vd_east_E.log  %log_path%\SUCCESS\det_r50_vd_east_E.log
        echo   det_r50_vd_east,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of det_r50_vd_east successfully!
)
python tools/infer_det.py -c configs/det/det_r50_vd_east.yml -o Global.use_gpu=False TestReader.infer_img="./doc/imgs_en/" Global.checkpoints="output/det_east/best_accuracy" > %log_path%/det_r50_vd_east_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\det_r50_vd_east_I.log  %log_path%\FAIL\det_r50_vd_east_I.log
        echo   det_r50_vd_east,infer,FAIL  >> %log_path%\result.log
        echo   infer of det_r50_vd_east failed!
) else (
        move  %log_path%\det_r50_vd_east_I.log  %log_path%\SUCCESS\det_r50_vd_east_I.log
        echo   det_r50_vd_east,infer,SUCCISS  >> %log_path%\result.log
        echo   infer of det_r50_vd_east successfully!
)
python tools/export_model.py -c configs/det/det_r50_vd_east.yml -o Global.use_gpu=False Global.checkpoints="output/det_east/best_accuracy" Global.save_inference_dir="./inference_det/det_east" > %log_path%/det_r50_vd_east_export.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\det_r50_vd_east_export.log  %log_path%\FAIL\det_r50_vd_east_export.log
        echo   det_r50_vd_east,export model,FAIL  >> %log_path%\result.log
        echo   export model of det_r50_vd_east failed!
) else (
        move  %log_path%\det_r50_vd_east_export.log  %log_path%\SUCCESS\det_r50_vd_east_export.log
        echo   det_r50_vd_east,export model,SUCCISS  >> %log_path%\result.log
        echo   export model of det_r50_vd_east successfully!
)
python tools/infer/predict_det.py --image_dir="./doc/imgs_en/img_10.jpg" --det_model_dir="./inference_det/det_east" --det_algorithm=EAST --use_gpu=False --use_tensorrt=False > %log_path%/det_r50_vd_east_P.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\det_r50_vd_east_P.log  %log_path%\FAIL\det_r50_vd_east_P.log
        echo   det_r50_vd_east,predict,FAIL  >> %log_path%\result.log
        echo   predict of det_r50_vd_east failed!
) else (
        move  %log_path%\det_r50_vd_east_P.log  %log_path%\SUCCESS\det_r50_vd_east_P.log
        echo   det_r50_vd_east,predict,SUCCISS  >> %log_path%\result.log
        echo   predict of det_r50_vd_east successfully!
)

set sed="C:\Program Files\Git\usr\bin\sed.exe"
rem rec rec_icdar15_train CRNN
python tools/train.py -c configs/rec/rec_icdar15_train.yml -o Global.use_gpu=False Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=10 Global.print_batch_step=10 > %log_path%/rec_icdar15_train_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_icdar15_train_T.log  %log_path%\FAIL\rec_icdar15_train_T.log
        echo   rec_icdar15_train,train,FAIL  >> %log_path%\result.log
        echo   train of rec_icdar15_train failed!
) else (
        move  %log_path%\rec_icdar15_train_T.log  %log_path%\SUCCESS\rec_icdar15_train_T.log
        echo   rec_icdar15_train,train,SUCCISS  >> %log_path%\result.log
        echo   train of rec_icdar15_train successfully!
)
python tools/eval.py -c configs/rec/rec_icdar15_train.yml  -o Global.use_gpu=False Global.checkpoints="output/rec_CRNN/best_accuracy" > %log_path%/rec_icdar15_train_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_icdar15_train_E.log  %log_path%\FAIL\rec_icdar15_train_E.log
        echo   rec_icdar15_train,eval,FAIL  >> %log_path%\result.log
        echo   eval of rec_icdar15_train failed!
) else (
        move  %log_path%\rec_icdar15_train_E.log  %log_path%\SUCCESS\rec_icdar15_train_E.log
        echo   rec_icdar15_train,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of rec_icdar15_train successfully!
)
python tools/infer_rec.py -c configs/rec/rec_icdar15_train.yml  -o Global.use_gpu=False Global.checkpoints="output/rec_CRNN/best_accuracy" TestReader.infer_img=doc/imgs_words/en/word_1.png > %log_path%/rec_icdar15_train_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_icdar15_train_I.log  %log_path%\FAIL\rec_icdar15_train_I.log
        echo   rec_icdar15_train,infer,FAIL  >> %log_path%\result.log
        echo   infer of rec_icdar15_train failed!
) else (
        move  %log_path%\rec_icdar15_train_I.log  %log_path%\SUCCESS\rec_icdar15_train_I.log
        echo   rec_icdar15_train,infer,SUCCISS  >> %log_path%\result.log
        echo   infer of rec_icdar15_train successfully!
)
python tools/export_model.py -c configs/rec/rec_icdar15_train.yml -o Global.use_gpu=False  Global.checkpoints="output/rec_CRNN/best_accuracy" Global.save_inference_dir=./inference_rec/rec_CRNN > %log_path%/rec_icdar15_train_export.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_icdar15_train_export.log  %log_path%\FAIL\rec_icdar15_train_export.log
        echo   rec_icdar15_train,export model,FAIL  >> %log_path%\result.log
        echo   export model of rec_icdar15_train failed!
) else (
        move  %log_path%\rec_icdar15_train_export.log  %log_path%\SUCCESS\rec_icdar15_train_export.log
        echo   rec_icdar15_train,export model,SUCCISS  >> %log_path%\result.log
        echo   export model of rec_icdar15_train successfully!
)
python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir=./inference_rec/rec_CRNN --rec_image_shape="3, 32, 100" --rec_char_type="en" --use_gpu=False --use_tensorrt=False > %log_path%/rec_icdar15_train_P.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_icdar15_train_P.log  %log_path%\FAIL\rec_icdar15_train_P.log
        echo   rec_icdar15_train,predict,FAIL  >> %log_path%\result.log
        echo   predict of rec_icdar15_train failed!
) else (
        move  %log_path%\rec_icdar15_train_P.log  %log_path%\SUCCESS\rec_icdar15_train_P.log
        echo   rec_icdar15_train,predict,SUCCISS  >> %log_path%\result.log
        echo   predict of rec_icdar15_train successfully!
)

rem rec_chinese_lite_train CRNN
%sed% -i s/"rec_chinese_reader.yml"/"rec_icdar15_reader.yml"/g configs/rec/rec_chinese_lite_train.yml
%sed% -i s/"ppocr_keys_v1.txt"/"ic15_dict.txt"/g configs/rec/rec_chinese_lite_train.yml
python tools/train.py -c configs/rec/rec_chinese_lite_train.yml -o Global.use_gpu=False Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=10 Global.print_batch_step=10 > %log_path%/rec_chinese_lite_train_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_chinese_lite_train_T.log  %log_path%\FAIL\rec_chinese_lite_train_T.log
        echo   rec_chinese_lite_train,train,FAIL  >> %log_path%\result.log
        echo   train of rec_chinese_lite_train failed!
) else (
        move  %log_path%\rec_chinese_lite_train_T.log  %log_path%\SUCCESS\rec_chinese_lite_train_T.log
        echo   rec_chinese_lite_train,train,SUCCISS  >> %log_path%\result.log
        echo   train of rec_chinese_lite_train successfully!
)
python tools/eval.py -c configs/rec/rec_chinese_lite_train.yml  -o Global.use_gpu=False Global.checkpoints="output/rec_CRNN/best_accuracy" > %log_path%/rec_chinese_lite_train_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_chinese_lite_train_E.log  %log_path%\FAIL\rec_chinese_lite_train_E.log
        echo   rec_chinese_lite_train,eval,FAIL  >> %log_path%\result.log
        echo   eval of rec_chinese_lite_train failed!
) else (
        move  %log_path%\rec_chinese_lite_train_E.log  %log_path%\SUCCESS\rec_chinese_lite_train_E.log
        echo   rec_chinese_lite_train,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of rec_chinese_lite_train successfully!
)
python tools/infer_rec.py -c configs/rec/rec_chinese_lite_train.yml  -o Global.use_gpu=False Global.checkpoints="output/rec_CRNN/best_accuracy" TestReader.infer_img=doc/imgs_words/en/word_1.png > %log_path%/rec_chinese_lite_train_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_chinese_lite_train_I.log  %log_path%\FAIL\rec_chinese_lite_train_I.log
        echo   rec_chinese_lite_train,infer,FAIL  >> %log_path%\result.log
        echo   infer of rec_chinese_lite_train failed!
) else (
        move  %log_path%\rec_chinese_lite_train_I.log  %log_path%\SUCCESS\rec_chinese_lite_train_I.log
        echo   rec_chinese_lite_train,infer,SUCCISS  >> %log_path%\result.log
        echo   infer of rec_chinese_lite_train successfully!
)
python tools/export_model.py -c configs/rec/rec_chinese_lite_train.yml -o Global.use_gpu=False  Global.checkpoints="output/rec_CRNN/best_accuracy" Global.save_inference_dir=./inference_rec/rec_CRNN > %log_path%/rec_chinese_lite_train_export.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_chinese_lite_train_export.log  %log_path%\FAIL\rec_chinese_lite_train_export.log
        echo   rec_chinese_lite_train,export model,FAIL  >> %log_path%\result.log
        echo   export model of rec_chinese_lite_train failed!
) else (
        move  %log_path%\rec_chinese_lite_train_export.log  %log_path%\SUCCESS\rec_chinese_lite_train_export.log
        echo   rec_chinese_lite_train,export model,SUCCISS  >> %log_path%\result.log
        echo   export model of rec_chinese_lite_train successfully!
)
python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir=./inference_rec/rec_CRNN --rec_image_shape="3, 32, 100" --rec_char_type="en" --use_gpu=False --use_tensorrt=False > %log_path%/rec_chinese_lite_train_P.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_chinese_lite_train_P.log  %log_path%\FAIL\rec_chinese_lite_train_P.log
        echo   rec_chinese_lite_train,predict,FAIL  >> %log_path%\result.log
        echo   predict of rec_chinese_lite_train failed!
) else (
        move  %log_path%\rec_chinese_lite_train_P.log  %log_path%\SUCCESS\rec_chinese_lite_train_P.log
        echo   rec_chinese_lite_train,predict,SUCCISS  >> %log_path%\result.log
        echo   predict of rec_chinese_lite_train successfully!
)

rem rec_r34_vd_none_none_ctc Rosetta
%sed% -i s/"rec_benchmark_reader.yml"/"rec_icdar15_reader.yml"/g configs/rec/rec_r34_vd_none_none_ctc.yml
python tools/train.py -c configs/rec/rec_r34_vd_none_none_ctc.yml -o Global.use_gpu=False Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=10 Global.print_batch_step=10 > %log_path%/rec_r34_vd_none_none_ctc_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_r34_vd_none_none_ctc_T.log  %log_path%\FAIL\rec_r34_vd_none_none_ctc_T.log
        echo   rec_r34_vd_none_none_ctc,train,FAIL  >> %log_path%\result.log
        echo   train of rec_r34_vd_none_none_ctc failed!
) else (
        move  %log_path%\rec_r34_vd_none_none_ctc_T.log  %log_path%\SUCCESS\rec_r34_vd_none_none_ctc_T.log
        echo   rec_r34_vd_none_none_ctc,train,SUCCISS  >> %log_path%\result.log
        echo   train of rec_r34_vd_none_none_ctc successfully!
)
python tools/eval.py -c configs/rec/rec_r34_vd_none_none_ctc.yml  -o Global.use_gpu=False Global.checkpoints="output/rec_Rosetta/best_accuracy" > %log_path%/rec_r34_vd_none_none_ctc_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_r34_vd_none_none_ctc_E.log  %log_path%\FAIL\rec_r34_vd_none_none_ctc_E.log
        echo   rec_r34_vd_none_none_ctc,eval,FAIL  >> %log_path%\result.log
        echo   eval of rec_r34_vd_none_none_ctc failed!
) else (
        move  %log_path%\rec_r34_vd_none_none_ctc_E.log  %log_path%\SUCCESS\rec_r34_vd_none_none_ctc_E.log
        echo   rec_r34_vd_none_none_ctc,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of rec_r34_vd_none_none_ctc successfully!
)
python tools/infer_rec.py -c configs/rec/rec_r34_vd_none_none_ctc.yml  -o Global.use_gpu=False Global.checkpoints="output/rec_Rosetta/best_accuracy" TestReader.infer_img=doc/imgs_words/en/word_1.png > %log_path%/rec_r34_vd_none_none_ctc_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_r34_vd_none_none_ctc_I.log  %log_path%\FAIL\rec_r34_vd_none_none_ctc_I.log
        echo   rec_r34_vd_none_none_ctc,infer,FAIL  >> %log_path%\result.log
        echo   infer of rec_r34_vd_none_none_ctc failed!
) else (
        move  %log_path%\rec_r34_vd_none_none_ctc_I.log  %log_path%\SUCCESS\rec_r34_vd_none_none_ctc_I.log
        echo   rec_r34_vd_none_none_ctc,infer,SUCCISS  >> %log_path%\result.log
        echo   infer of rec_r34_vd_none_none_ctc successfully!
)
python tools/export_model.py -c configs/rec/rec_r34_vd_none_none_ctc.yml -o Global.use_gpu=False  Global.checkpoints="output/rec_Rosetta/best_accuracy" Global.save_inference_dir=./inference_rec/rec_Rosetta > %log_path%/rec_r34_vd_none_none_ctc_export.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_r34_vd_none_none_ctc_export.log  %log_path%\FAIL\rec_r34_vd_none_none_ctc_export.log
        echo   rec_r34_vd_none_none_ctc,export model,FAIL  >> %log_path%\result.log
        echo   export model of rec_r34_vd_none_none_ctc failed!
) else (
        move  %log_path%\rec_r34_vd_none_none_ctc_export.log  %log_path%\SUCCESS\rec_r34_vd_none_none_ctc_export.log
        echo   rec_r34_vd_none_none_ctc,export model,SUCCISS  >> %log_path%\result.log
        echo   export model of rec_r34_vd_none_none_ctc successfully!
)
python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir=./inference_rec/rec_Rosetta --rec_image_shape="3, 32, 100" --rec_char_type="en" --use_gpu=False --use_tensorrt=False > %log_path%/rec_r34_vd_none_none_ctc_P.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_r34_vd_none_none_ctc_P.log  %log_path%\FAIL\rec_r34_vd_none_none_ctc_P.log
        echo   rec_r34_vd_none_none_ctc,predict,FAIL  >> %log_path%\result.log
        echo   predict of rec_r34_vd_none_none_ctc failed!
) else (
        move  %log_path%\rec_r34_vd_none_none_ctc_P.log  %log_path%\SUCCESS\rec_r34_vd_none_none_ctc_P.log
        echo   rec_r34_vd_none_none_ctc,predict,SUCCISS  >> %log_path%\result.log
        echo   predict of rec_r34_vd_none_none_ctc successfully!
)

rem rec_mv3_tps_bilstm_ctc STARNet
%sed% -i s/"rec_benchmark_reader.yml"/"rec_icdar15_reader.yml"/g configs/rec/rec_mv3_tps_bilstm_ctc.yml
python tools/train.py -c configs/rec/rec_mv3_tps_bilstm_ctc.yml -o Global.use_gpu=False Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=10 Global.print_batch_step=10 > %log_path%/rec_mv3_tps_bilstm_ctc_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_mv3_tps_bilstm_ctc_T.log  %log_path%\FAIL\rec_mv3_tps_bilstm_ctc_T.log
        echo   rec_mv3_tps_bilstm_ctc,train,FAIL  >> %log_path%\result.log
        echo   train of rec_mv3_tps_bilstm_ctc failed!
) else (
        move  %log_path%\rec_mv3_tps_bilstm_ctc_T.log  %log_path%\SUCCESS\rec_mv3_tps_bilstm_ctc_T.log
        echo   rec_mv3_tps_bilstm_ctc,train,SUCCISS  >> %log_path%\result.log
        echo   train of rec_mv3_tps_bilstm_ctc successfully!
)
python tools/eval.py -c configs/rec/rec_mv3_tps_bilstm_ctc.yml  -o Global.use_gpu=False Global.checkpoints="output/rec_STARNet/best_accuracy" > %log_path%/rec_mv3_tps_bilstm_ctc_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_mv3_tps_bilstm_ctc_E.log  %log_path%\FAIL\rec_mv3_tps_bilstm_ctc_E.log
        echo   rec_mv3_tps_bilstm_ctc,eval,FAIL  >> %log_path%\result.log
        echo   eval of rec_mv3_tps_bilstm_ctc failed!
) else (
        move  %log_path%\rec_mv3_tps_bilstm_ctc_E.log  %log_path%\SUCCESS\rec_mv3_tps_bilstm_ctc_E.log
        echo   rec_mv3_tps_bilstm_ctc,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of rec_mv3_tps_bilstm_ctc successfully!
)
python tools/infer_rec.py -c configs/rec/rec_mv3_tps_bilstm_ctc.yml  -o Global.use_gpu=False Global.checkpoints="output/rec_STARNet/best_accuracy" TestReader.infer_img=doc/imgs_words/en/word_1.png > %log_path%/rec_mv3_tps_bilstm_ctc_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_mv3_tps_bilstm_ctc_I.log  %log_path%\FAIL\rec_mv3_tps_bilstm_ctc_I.log
        echo   rec_mv3_tps_bilstm_ctc,infer,FAIL  >> %log_path%\result.log
        echo   infer of rec_mv3_tps_bilstm_ctc failed!
) else (
        move  %log_path%\rec_mv3_tps_bilstm_ctc_I.log  %log_path%\SUCCESS\rec_mv3_tps_bilstm_ctc_I.log
        echo   rec_mv3_tps_bilstm_ctc,infer,SUCCISS  >> %log_path%\result.log
        echo   infer of rec_mv3_tps_bilstm_ctc successfully!
)
python tools/export_model.py -c configs/rec/rec_mv3_tps_bilstm_ctc.yml -o Global.use_gpu=False  Global.checkpoints="output/rec_STARNet/best_accuracy" Global.save_inference_dir=./inference_rec/rec_STARNet > %log_path%/rec_mv3_tps_bilstm_ctc_export.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_mv3_tps_bilstm_ctc_export.log  %log_path%\FAIL\rec_mv3_tps_bilstm_ctc_export.log
        echo   rec_mv3_tps_bilstm_ctc,export model,FAIL  >> %log_path%\result.log
        echo   export model of rec_mv3_tps_bilstm_ctc failed!
) else (
        move  %log_path%\rec_mv3_tps_bilstm_ctc_export.log  %log_path%\SUCCESS\rec_mv3_tps_bilstm_ctc_export.log
        echo   rec_mv3_tps_bilstm_ctc,export model,SUCCISS  >> %log_path%\result.log
        echo   export model of rec_mv3_tps_bilstm_ctc successfully!
)
python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir=./inference_rec/rec_STARNet --rec_image_shape="3, 32, 100" --rec_char_type="en" --use_gpu=False --use_tensorrt=False > %log_path%/rec_mv3_tps_bilstm_ctc_P.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_mv3_tps_bilstm_ctc_P.log  %log_path%\FAIL\rec_mv3_tps_bilstm_ctc_P.log
        echo   rec_mv3_tps_bilstm_ctc,predict,FAIL  >> %log_path%\result.log
        echo   predict of rec_mv3_tps_bilstm_ctc failed!
) else (
        move  %log_path%\rec_mv3_tps_bilstm_ctc_P.log  %log_path%\SUCCESS\rec_mv3_tps_bilstm_ctc_P.log
        echo   rec_mv3_tps_bilstm_ctc,predict,SUCCISS  >> %log_path%\result.log
        echo   predict of rec_mv3_tps_bilstm_ctc successfully!
)

rem rec_mv3_tps_bilstm_attn
%sed% -i s/"rec_benchmark_reader.yml"/"rec_icdar15_reader.yml"/g configs/rec/rec_mv3_tps_bilstm_attn.yml
python tools/train.py -c configs/rec/rec_mv3_tps_bilstm_attn.yml -o Global.use_gpu=False Global.epoch_num=1 Global.save_epoch_step=1 Global.eval_batch_step=10 Global.print_batch_step=10 > %log_path%/rec_mv3_tps_bilstm_attn_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_mv3_tps_bilstm_attn_T.log  %log_path%\FAIL\rec_mv3_tps_bilstm_attn_T.log
        echo   rec_mv3_tps_bilstm_attn,train,FAIL  >> %log_path%\result.log
        echo   train of rec_mv3_tps_bilstm_attn failed!
) else (
        move  %log_path%\rec_mv3_tps_bilstm_attn_T.log  %log_path%\SUCCESS\rec_mv3_tps_bilstm_attn_T.log
        echo   rec_mv3_tps_bilstm_attn,train,SUCCISS  >> %log_path%\result.log
        echo   train of rec_mv3_tps_bilstm_attn successfully!
)
python tools/eval.py -c configs/rec/rec_mv3_tps_bilstm_attn.yml  -o Global.use_gpu=False Global.checkpoints="output/rec_RARE/best_accuracy" > %log_path%/rec_mv3_tps_bilstm_attn_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_mv3_tps_bilstm_attn_E.log  %log_path%\FAIL\rec_mv3_tps_bilstm_attn_E.log
        echo   rec_mv3_tps_bilstm_attn,eval,FAIL  >> %log_path%\result.log
        echo   eval of rec_mv3_tps_bilstm_attn failed!
) else (
        move  %log_path%\rec_mv3_tps_bilstm_attn_E.log  %log_path%\SUCCESS\rec_mv3_tps_bilstm_attn_E.log
        echo   rec_mv3_tps_bilstm_attn,eval,SUCCISS  >> %log_path%\result.log
        echo   eval of rec_mv3_tps_bilstm_attn successfully!
)
python tools/infer_rec.py -c configs/rec/rec_mv3_tps_bilstm_attn.yml  -o Global.use_gpu=False Global.checkpoints="output/rec_RARE/best_accuracy" TestReader.infer_img=doc/imgs_words/en/word_1.png > %log_path%/rec_mv3_tps_bilstm_attn_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_mv3_tps_bilstm_attn_I.log  %log_path%\FAIL\rec_mv3_tps_bilstm_attn_I.log
        echo   rec_mv3_tps_bilstm_attn,infer,FAIL  >> %log_path%\result.log
        echo   infer of rec_mv3_tps_bilstm_attn failed!
) else (
        move  %log_path%\rec_mv3_tps_bilstm_attn_I.log  %log_path%\SUCCESS\rec_mv3_tps_bilstm_attn_I.log
        echo   rec_mv3_tps_bilstm_attn,infer,SUCCISS  >> %log_path%\result.log
        echo   infer of rec_mv3_tps_bilstm_attn successfully!
)
python tools/export_model.py -c configs/rec/rec_mv3_tps_bilstm_attn.yml -o Global.use_gpu=False  Global.checkpoints="output/rec_RARE/best_accuracy" Global.save_inference_dir=./inference_rec/rec_RARE > %log_path%/rec_mv3_tps_bilstm_attn_export.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_mv3_tps_bilstm_attn_export.log  %log_path%\FAIL\rec_mv3_tps_bilstm_attn_export.log
        echo   rec_mv3_tps_bilstm_attn,export model,FAIL  >> %log_path%\result.log
        echo   export model of rec_mv3_tps_bilstm_attn failed!
) else (
        move  %log_path%\rec_mv3_tps_bilstm_attn_export.log  %log_path%\SUCCESS\rec_mv3_tps_bilstm_attn_export.log
        echo   rec_mv3_tps_bilstm_attn,export model,SUCCISS  >> %log_path%\result.log
        echo   export model of rec_mv3_tps_bilstm_attn successfully!
)
python tools/infer/predict_rec.py --image_dir="./doc/imgs_words_en/word_336.png" --rec_model_dir=./inference_rec/rec_RARE --rec_image_shape="3, 32, 100" --rec_char_type="en" --use_gpu=False --use_tensorrt=False > %log_path%/rec_mv3_tps_bilstm_attn_P.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\rec_mv3_tps_bilstm_attn_P.log  %log_path%\FAIL\rec_mv3_tps_bilstm_attn_P.log
        echo   rec_mv3_tps_bilstm_attn,predict,FAIL  >> %log_path%\result.log
        echo   predict of rec_mv3_tps_bilstm_attn failed!
) else (
        move  %log_path%\rec_mv3_tps_bilstm_attn_P.log  %log_path%\SUCCESS\rec_mv3_tps_bilstm_attn_P.log
        echo   rec_mv3_tps_bilstm_attn,predict,SUCCISS  >> %log_path%\result.log
        echo   predict of rec_mv3_tps_bilstm_attn successfully!
)

rem system 
python tools/infer/predict_system.py --image_dir="./doc/imgs/2.jpg" --det_model_dir="./models/inference/det/"  --rec_model_dir="./models/inference/rec/" > %log_path%/system_predict.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\system_predict.log  %log_path%\FAIL\system_predict.log
        echo   system_predict,predict,FAIL  >> %log_path%\result.log
        echo   predict of system_predict failed!
) else (
        move  %log_path%\system_predict.log  %log_path%\SUCCESS\system_predict.log
        echo   system_predict,predict,SUCCISS  >> %log_path%\result.log
        echo   predict of system_predict successfully!
)

