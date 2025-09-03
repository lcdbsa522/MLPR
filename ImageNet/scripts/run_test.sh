####################################################################################
# You may run the code using the following commands.
# Please change the directory for the dataset (i.e., /path/to/ILSVRC2012).
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.
####################################################################################

# ####### ResNet-18 W4A4 Multi Quantizer
# python ImageNet_train_quant.py --data "../../ImageNet" \
#                                --visible_gpus '0, 1' \
#                                --workers 10 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --batch-size 256 \
#                                --baseline False \
#                                --use_hessian True \
#                                --update_scales_every 1 \
#                                --weight_levels 16 \
#                                --act_levels 16 \
#                                --feature_levels 16 \
#                                --QFeatureFlag True \
#                                --use_student_quant_params True \
#                                --use_adapter True \
#                                --use_adapter_bn False \
#                                --use_adapter_scaling_factor False \
#                                --use_multi_quantizer True \
#                                --middle_feature_levels 256 \
#                                --distill 'fd' \
#                                --distill_loss 'L2' \
#                                --teacher_arch 'resnet18_fp' \
#                                --kd_beta 0.0 \
#                                --kd_gamma 1.0 \
#                                --kd_alpha 1.0 \
#                                --log_dir "./results/ResNet18_ImageNet/test/multi_quantizer_W4A4/"


# ####### ResNet-18 W4A4 Scalar Scaling Factor
# python ImageNet_train_quant.py --data "/mnt/1TBSSD/imagenet_cls_loc_pytorch" \
#                                --visible_gpus '0' \
#                                --workers 10 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --batch-size 128 \
#                                --baseline False \
#                                --use_hessian True \
#                                --update_scales_every 1 \
#                                --weight_levels 16 \
#                                --act_levels 16 \
#                                --feature_levels 16 \
#                                --QFeatureFlag True \
#                                --use_student_quant_params True \
#                                --use_adapter True \
#                                --use_adapter_bn True \
#                                --use_adapter_scaling_factor True \
#                                --adapter_scaling_factor_type 'scalar' \
#                                --use_multi_quantizer False \
#                                --distill 'fd' \
#                                --distill_loss 'L2' \
#                                --teacher_arch 'resnet18_fp' \
#                                --kd_beta 0.0 \
#                                --kd_gamma 1.0 \
#                                --kd_alpha 1.0 \
#                                --log_dir "./results/ResNet18_ImageNet/test/scalar_scaling_factor_W4A4/" \
#                                --resume "./results/ResNet18_ImageNet/test/scalar_scaling_factor_W4A4/checkpoint.pth.tar"

####### ResNet-18 W4A4 Vector Scaling Factor
python ImageNet_train_quant.py --data "../ILSVRC2012" \
                               --visible_gpus '0' \
                               --workers 10 \
                               --arch 'resnet18_quant' \
                               --epochs 100 \
                               --batch-size 128 \
                               --baseline False \
                               --use_hessian True \
                               --update_scales_every 1 \
                               --weight_levels 16 \
                               --act_levels 16 \
                               --feature_levels 16 \
                               --QFeatureFlag True \
                               --use_student_quant_params True \
                               --use_adapter True \
                               --use_adapter_bn False \
                               --use_adapter_scaling_factor True \
                               --adapter_scaling_factor_type 'vector' \
                               --use_multi_quantizer False \
                               --distill 'fd' \
                               --distill_loss 'L2' \
                               --teacher_arch 'resnet18_fp' \
                               --kd_beta 0.0 \
                               --kd_gamma 1.0 \
                               --kd_alpha 1.0 \
                               --log_dir "./results/ResNet18_ImageNet/test/vector_scaling_factor_W4A4/"