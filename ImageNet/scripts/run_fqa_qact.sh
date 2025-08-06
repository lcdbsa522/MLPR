####################################################################################
# You may run the code using the following commands.
# Please change the directory for the dataset (i.e., /path/to/ILSVRC2012).
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.
####################################################################################

####### ResNet-18 W1A1 ours
python ImageNet_train_quant.py --data "../../ImageNet" \
                               --gpu 0 \
                               --visible_gpus "0" \
                               --workers 16 \
                               --arch "resnet18_quant" \
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
                               --distill_loss "L2" \
                               --distill "fd" \
                               --teacher_arch "resnet18_fp" \
                               --kd_beta 1.0 \
                               --kd_gamma 0.0 \
                               --kd_alpha 0.5 \
                               --log_dir "./results/ResNet18_ImageNet/W4A4/FQA_Qact_L2_beta1.0_gamma0_alpha0.5"

####### ResNet-18 W1A1 ours (fixed scaling factor)
# python ImageNet_train_quant.py --data "/path/to/ILSVRC2012" \
#                                --dist-url 'tcp://127.0.0.1:23457' \
#                                --visible_gpus '1,3,5,7' \
#                                --workers 20 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --weight_levels 2 \
#                                --act_levels 2 \
#                                --baseline False \
#                                --use_hessian False \
#                                --bkwd_scaling_factorW 0.001 \
#                                --bkwd_scaling_factorA 0.001 \
#                                --log_dir "../results/ResNet18/ours(fix)/W1A1_0.001/"

####### ResNet-18 W1A1 baseline (STE)
# python ImageNet_train_quant.py --data "/path/to/ILSVRC2012" \
#                                --visible_gpus '0,1,2,3' \
#                                --workers 20 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --weight_levels 2 \
#                                --act_levels 2 \
#                                --baseline True \
#                                --log_dir "../results/ResNet18/base/W1A1/"

####### ResNet-18 W1A32 ours
# python ImageNet_train_quant.py --data "/path/to/ILSVRC2012" \
#                                --visible_gpus '0,2,4,6' \
#                                --workers 20 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --weight_levels 2 \
#                                --QActFlag False \
#                                --baseline False \
#                                --use_hessian True \
#                                --update_scales_every 1 \
#                                --log_dir "../results/ResNet18/ours(hess)/W1A32/"

####### ResNet-18 W1A32 baseline (STE)
# python ImageNet_train_quant.py --data "/path/to/ILSVRC2012" \
#                                --dist-url 'tcp://127.0.0.1:23457' \
#                                --visible_gpus '1,3,5,7' \
#                                --workers 20 \
#                                --arch 'resnet18_quant' \
#                                --epochs 100 \
#                                --weight_levels 2 \
#                                --QActFlag False \
#                                --baseline True \
#                                --log_dir "../results/ResNet18/base/W1A32/"