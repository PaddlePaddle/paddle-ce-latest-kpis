@echo off
rem This file is only used for continuous evaluation.

set CUDA_VISIBLE_DEVICES=0
	
rem distill
cd demo/distillation
python distill.py --num_epochs 1 --batch_size 32 --data imagenet > distill.log 2>&1
type distill.log |grep train_epoch|awk   -F "[, ]" "END{print \"kpis\tdistillation_train_loss\t\"$12}" | python _ce.py
type distill.log |grep valid|awk -F "[, ]" "END{print \"kpis\tdistillation_valid_loss\t\"$9}" | python _ce.py


rem quant_aware
cd %~dp0
cd demo/quant/quant_aware
if not exist pretrain ( 
md pretrain
cd pretrain
mklink /j MobileNetV1_pretrained %data_path%\PaddleSlim\pretrained\MobileNetV1_pretrained
cd ..
)
python train.py --model MobileNet --pretrained_model ./pretrain/MobileNetV1_pretrained --checkpoint_dir ./output/mobilenetv1 --num_epochs 1 --data imagenet --batch_size 32 >quant_aware.log 2>&1
type quant_aware.log 2>&1|grep loss | awk -F "[:;]" "END{print \"kpis\tprune_train_loss\t\"$5}"|python _ce.pys

rem quant_embedding
cd %~dp0
cd demo/quant/quant_embedding
if not exist data (
mklink /j data %data_path%\word2vec
)
set OPENBLAS_NUM_THREADS=1 
set CPU_NUM=1
python train.py --train_data_dir data/convert_text8 --dict_path data/test_build_dict --num_passes 1 --batch_size 100 --model_output_dir v1_cpu5_b100_lr1dir --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse > %log_path%/slim_clssification_quant_embed_train.log
if %errorlevel% GTR 0 (
        move  %log_path%\slim_clssification_quant_embed_train.log  %log_path%\FAIL\slim_clssification_quant_embed_train.log
        echo   slim_clssification_quant_embed,train,FAIL  >> %log_path%\result.log
        echo   train of slim_clssification_quant_embed failed!
) else (
        move  %log_path%\slim_clssification_quant_embed_train.log  %log_path%\SUCCESS\slim_clssification_quant_embed_train.log
        echo   slim_clssification_quant_embed,train,SUCCESS  >> %log_path%\result.log
        echo   train of slim_clssification_quant_embed successfully!
 )
python infer.py --infer_epoch --test_dir data/test_mid_dir --dict_path data/test_build_dict_word_to_id_ --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 --last_index 0 > %log_path%/slim_clssification_quant_embed_before_infer.log
if %errorlevel% GTR 0 (
        move  %log_path%\slim_clssification_quant_embed_before_infer.log  %log_path%\FAIL\slim_clssification_quant_embed_before_infer.log
        echo   slim_clssification_quant_embed_before,infer,FAIL  >> %log_path%\result.log
        echo   infer of slim_clssification_quant_embed_before failed!
) else (
        move  %log_path%\slim_clssification_quant_embed_before_infer.log  %log_path%\SUCCESS\slim_clssification_quant_embed_before_infer.log
        echo   slim_clssification_quant_embed_before,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of slim_clssification_quant_embed_before successfully!
 )
python infer.py --infer_epoch --test_dir data/test_mid_dir --dict_path data/test_build_dict_word_to_id_ --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  --start_index 0 --last_index 0 --emb_quant True > %log_path%/slim_clssification_quant_embed_after_infer.log
if %errorlevel% GTR 0 (
        move  %log_path%\slim_clssification_quant_embed_after_infer.log  %log_path%\FAIL\slim_clssification_quant_embed_after_infer.log
        echo   slim_clssification_quant_embed_after,infer,FAIL  >> %log_path%\result.log
        echo   infer of slim_clssification_quant_embed_after failed!
) else (
        move  %log_path%\slim_clssification_quant_embed_after_infer.log  %log_path%\SUCCESS\slim_clssification_quant_embed_after_infer.log
        echo   slim_clssification_quant_embed_after,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of slim_clssification_quant_embed_after successfully!
 )

rem quant post
cd %~dp0
cd demo/quant/quant_post
if not exist pretrain ( 
md pretrain
cd pretrain
mklink /j MobileNetV1_pretrained %data_path%\PaddleSlim\pretrained\MobileNetV1_pretrained
cd ..
)
python export_model.py --model "MobileNet" --pretrained_model ./pretrain/MobileNetV1_pretrained --data imagenet > %log_path%/slim_clssificaiton_quant_post_export_model.log
if %errorlevel% GTR 0 (
        move  %log_path%\slim_clssificaiton_quant_post_export_model.log  %log_path%\FAIL\slim_clssificaiton_quant_post_export_model.log
        echo   slim_clssificaiton_quant_post,export_model,FAIL  >> %log_path%\result.log
        echo   export_model of slim_clssificaiton_quant_post failed!
) else (
        move  %log_path%\slim_clssificaiton_quant_post_export_model.log  %log_path%\SUCCESS\slim_clssificaiton_quant_post_export_model.log
        echo   slim_clssificaiton_quant_post,export_model,SUCCESS  >> %log_path%\result.log
        echo   export_model of slim_clssificaiton_quant_post successfully!
 )
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fraction_of_gpu_memory_to_use=0.98
python quant_post.py --model_path ./inference_model/MobileNet --save_path ./quant_model_train/MobileNet --model_filename model --params_filename weights >  %log_path%/slim_clssificaiton_quant_post_train.log
if %errorlevel% GTR 0 (
        move  %log_path%\slim_clssificaiton_quant_post_train.log  %log_path%\FAIL\slim_clssificaiton_quant_post_train.log
        echo   slim_clssificaiton_quant_post,train,FAIL  >> %log_path%\result.log
        echo   train of slim_clssificaiton_quant_post failed!
) else (
        move  %log_path%\slim_clssificaiton_quant_post_train.log  %log_path%\SUCCESS\slim_clssificaiton_quant_post_train.log
        echo   slim_clssificaiton_quant_post,train,SUCCESS  >> %log_path%\result.log
        echo   train of slim_clssificaiton_quant_post successfully!
 )
python eval.py --model_path ./inference_model/MobileNet --model_name model --params_name weights > %log_path%/slim_clssificaiton_quant_post_eval.log
if %errorlevel% GTR 0 (
        move  %log_path%\slim_clssificaiton_quant_post_eval.log  %log_path%\FAIL\slim_clssificaiton_quant_post_eval.log
        echo   slim_clssificaiton_quant_post,eval,FAIL  >> %log_path%\result.log
        echo   eval of slim_clssificaiton_quant_post failed!
) else (
        move  %log_path%\slim_clssificaiton_quant_post_eval.log  %log_path%\SUCCESS\slim_clssificaiton_quant_post_eval.log
        echo   slim_clssificaiton_quant_post,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of slim_clssificaiton_quant_post successfully!
 )
python eval.py --model_path ./quant_model_train/MobileNet > %log_path%/slim_clssificaiton_quant_post_quant_eval.log
if %errorlevel% GTR 0 (
        move  %log_path%\slim_clssificaiton_quant_post_quant_eval.log  %log_path%\FAIL\slim_clssificaiton_quant_post_quant_eval.log
        echo   slim_clssificaiton_quant_post,quant_eval,FAIL  >> %log_path%\result.log
        echo   quant_eval of slim_clssificaiton_quant_post failed!
) else (
        move  %log_path%\slim_clssificaiton_quant_post_quant_eval.log  %log_path%\SUCCESS\slim_clssificaiton_quant_post_quant_eval.log
        echo   slim_clssificaiton_quant_post,quant_eval,SUCCESS  >> %log_path%\result.log
        echo   quant_eval of slim_clssificaiton_quant_post successfully!
 )
rem prune
cd %~dp0
cd demo/prune
if not exist pretrain ( 
md pretrain
cd pretrain
mklink /j MobileNetV1_pretrained %data_path%\PaddleSlim\pretrained\MobileNetV1_pretrained
cd ..
)
train.py --model "MobileNet" --pruned_ratio 0.31 --data "imagenet" --pretrained_model ./pretrain/MobileNetV1_pretrained/ --num_epochs 1 --batch_size 64 > prune.log.log
type prune.log|grep "loss"|awk -F "[;:]" "END{print \"kpis\tprune_loss\t\"$5}" | python _ce.py
python eval.py  --model "MobileNet"  --data "imagenet" --model_path "./models/0" > %log_path%/slim_clssification_prune_eval.log
if %errorlevel% GTR 0 (
        move  %log_path%\slim_clssification_prune_eval.log  %log_path%\FAIL\slim_clssification_prune_eval.log
        echo   slim_clssification_prune,eval,FAIL  >> %log_path%\result.log
        echo   eval of slim_clssification_prune failed!
) else (
        move  %log_path%\slim_clssification_prune_eval.log  %log_path%\SUCCESS\slim_clssification_prune_eval.log
        echo   slim_clssification_prune,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of slim_clssification_prune successfully!
 )
 
rem sa_nas_mobilenetv2
cd %~dp0
cd demo/nas
python sa_nas_mobilenetv2.py --search_steps 1 --port 8881  --data imagenet --retain_epoch 1 --batch_size 32 --use_gpu True > sa_nas_mobilenetv2.log 2>&1
type sa_nas_mobilenetv2.log| grep "FINAL TEST"|awk -F "[,:]" "END{print \"kpis\tnan_test_loss\t\"$7}" | python _ce.py
python block_sa_nas_mobilenetv2.py --search_steps 1 --port 8883 --data imagenet --retain_epoch 1 --use_gpu True > block_sa_nas_mobilenetv2.log 2>&1
type  block_sa_nas_mobilenetv2.log|grep "current_flops"|awk -F "[:]" "{print \"kpis\tnas_current_flops\t\"$3}" | python _ce.py








