@echo off
set CUDA_VISIBLE_DEVICES=0

set sed="C:\Program Files\Git\usr\bin\sed.exe"
rem cascase_rcnn
%sed% -i s/"base_lr: 0.02"/"base_lr: 0.0025"/g configs/cascade_rcnn_r50_fpn_1x.yml
python tools/train.py -c configs/cascade_rcnn_r50_fpn_1x.yml -o max_iters=200 FasterRCNNTrainFeed.shuffle=false CascadeBBoxAssigner.shuffle_before_sample=false FPNRPNHead.rpn_target_assign.use_random=false | python _ce.py
rem eval
python tools/eval.py -c configs/cascade_rcnn_r50_fpn_1x.yml -o use_gpu=True > %log_path%/cascade_rcnn_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\cascade_rcnn_E.log  %log_path%\FAIL\cascade_rcnn_E.log
        echo   cascade_rcnn,eval,FAIL  >> %log_path%\result.log
        echo   evaling of cascade_rcnn failed!
) else (
        move  %log_path%\cascade_rcnn_E.log  %log_path%\SUCCESS\cascade_rcnn_E.log
        echo   cascade_rcnn,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of cascade_rcnn successfully!
)
python tools/infer.py -c configs/cascade_rcnn_r50_fpn_1x.yml --infer_img=demo/000000570688.jpg -o use_gpu=True > %log_path%/cascade_rcnn_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\cascade_rcnn_I.log  %log_path%\FAIL\cascade_rcnn_I.log
        echo   cascade_rcnn,infer,FAIL  >> %log_path%\result.log
        echo   infering of cascade_rcnn failed!
) else (
        move  %log_path%\cascade_rcnn_I.log  %log_path%\SUCCESS\cascade_rcnn_I.log
        echo   cascade_rcnn,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of cascade_rcnn successfully!
)
rem faster_rcnn
%sed% -i s/"base_lr: 0.02"/"base_lr: 0.0025"/g configs/faster_rcnn_r50_fpn_1x.yml
python tools/train.py -c configs/faster_rcnn_r50_fpn_1x.yml -o max_iters=200 base_lr=0.02 FasterRCNNTrainFeed.shuffle=false | python _ce.py
rem eval
python tools/eval.py -c configs/faster_rcnn_r50_fpn_1x.yml -o use_gpu=True > %log_path%/faster_rcnn_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\faster_rcnn_E.log  %log_path%\FAIL\faster_rcnn_E.log
        echo   faster_rcnn,eval,FAIL  >> %log_path%\result.log
        echo   evaling of faster_rcnn failed!
) else (
        move  %log_path%\faster_rcnn_E.log  %log_path%\SUCCESS\faster_rcnn_E.log
        echo   faster_rcnn,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of faster_rcnn successfully!
)
python tools/infer.py -c configs/faster_rcnn_r50_fpn_1x.yml --infer_img=demo/000000570688.jpg -o use_gpu=True > %log_path%/faster_rcnn_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\faster_rcnn_I.log  %log_path%\FAIL\faster_rcnn_I.log
        echo   faster_rcnn,infer,FAIL  >> %log_path%\result.log
        echo   infering of faster_rcnn failed!
) else (
        move  %log_path%\faster_rcnn_I.log  %log_path%\SUCCESS\faster_rcnn_I.log
        echo   faster_rcnn,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of faster_rcnn successfully!
)

rem mask_rcnn
%sed% -i s/"base_lr: 0.01"/"base_lr: 0.00125"/g configs/mask_rcnn_r50_fpn_1x.yml
python tools/train.py -c configs/mask_rcnn_r50_fpn_1x.yml -o max_iters=200 MaskRCNNTrainFeed.shuffle=false BBoxAssigner.shuffle_before_sample=false FPNRPNHead.rpn_target_assign.use_random=false | python _ce.py
rem eval
python tools/eval.py -c configs/mask_rcnn_r50_fpn_1x.yml -o use_gpu=True > %log_path%/mask_rcnn_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\mask_rcnn_E.log  %log_path%\FAIL\mask_rcnn_E.log
        echo   mask_rcnn,eval,FAIL  >> %log_path%\result.log
        echo   evaling of mask_rcnn failed!
) else (
        move  %log_path%\mask_rcnn_E.log  %log_path%\SUCCESS\mask_rcnn_E.log
        echo   mask_rcnn,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of mask_rcnn successfully!
)
python tools/infer.py -c configs/mask_rcnn_r50_fpn_1x.yml --infer_img=demo/000000570688.jpg -o use_gpu=True > %log_path%/mask_rcnn_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\mask_rcnn_I.log  %log_path%\FAIL\mask_rcnn_I.log
        echo   mask_rcnn,infer,FAIL  >> %log_path%\result.log
        echo   infering of mask_rcnn failed!
) else (
        move  %log_path%\mask_rcnn_I.log  %log_path%\SUCCESS\mask_rcnn_I.log
        echo   mask_rcnn,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of mask_rcnn successfully!
)

rem yolov3
%sed% -i s/"batch_size: 8"/"batch_size: 4"/g configs/yolov3_darknet.yml
python tools/train.py -c configs/yolov3_darknet.yml -o max_iters=200 YoloTrainFeed.shuffle=false YoloTrainFeed.use_process=false | python _ce.py
rem eval
python tools/eval.py -c configs/yolov3_darknet.yml -o use_gpu=True > %log_path%/yolov3_darknet_E.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\yolov3_darknet_E.log  %log_path%\FAIL\yolov3_darknet_E.log
        echo   yolov3_darknet,eval,FAIL  >> %log_path%\result.log
        echo   evaling of yolov3_darknet failed!
) else (
        move  %log_path%\yolov3_darknet_E.log  %log_path%\SUCCESS\yolov3_darknet_E.log
        echo   yolov3_darknet,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of yolov3_darknet successfully!
)
python tools/infer.py -c configs/yolov3_darknet.yml --infer_img=demo/000000570688.jpg -o use_gpu=True > %log_path%/yolov3_darknet_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\yolov3_darknet_I.log  %log_path%\FAIL\yolov3_darknet_I.log
        echo   yolov3_darknet,infer,FAIL  >> %log_path%\result.log
        echo   infering of yolov3_darknet failed!
) else (
        move  %log_path%\yolov3_darknet_I.log  %log_path%\SUCCESS\yolov3_darknet_I.log
        echo   yolov3_darknet,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of yolov3_darknet successfully!
)


