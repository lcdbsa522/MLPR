#!/bin/bash


####################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
# 'weight_levels' and 'act_levels' correspond to 2^b, where b is a target bit-width.

# Method: FQA+EWGS
# Bit-width: T1, T2, T4, W1A1, W2A2, W4A4
####################################################################################

set -e

start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

# define methods and alpha
# METHOD_TYPE = "FQA_Qact_1_1bit/"
# ALPHA = 1
# GPU_ID = "0"
METHOD_TYPE=$1
ALPHA=$2
GPU_ID=$3

if [ -z "$METHOD_TYPE" ] || [ -z "$ALPHA" ]; then
    echo "Usage: $0 METHOD_TYPE ALPHA"
    echo "Example: $0 'FQA_Qact_L1_1bit_1' 1 '0'"
    exit 1
fi

echo "Running method: $METHOD_TYPE"
echo "KD_Alpha value: $ALPHA"
echo "GPU ID: $GPU_ID"

# set bit-widths based on method
if [[ $METHOD_TYPE == *"1bit"* ]]; then
    BIT_LEVEL=2
elif [[ $METHOD_TYPE == *"2bit"* ]]; then
    BIT_LEVEL=4
elif [[ $METHOD_TYPE == *"4bit"* ]]; then
    BIT_LEVEL=16
else
    echo "Unknown bit format in METHOD_TYPE: $METHOD_TYPE"
    exit 1
fi

# set distill loss based on method
if [[ $METHOD_TYPE == *"L1"* ]]; then
    DISTILL_LOSS='L1'
elif [[ $METHOD_TYPE == *"L2"* ]]; then
    DISTILL_LOSS='L2'
else
    echo "Unknown Loss format in METHOD_TYPE: $METHOD_TYPE"
    exit 1
fi

echo "BIT_LEVEL:  $BIT_LEVEL"
echo "Distillation Loss: $DISTILL_LOSS"

# train method
python3 train_quant_distill.py --gpu_id $GPU_ID \
                --arch 'resnet20_quant' \
                --teacher_arch 'resnet20_fp' \
                --weight_levels $BIT_LEVEL \
                --act_levels $BIT_LEVEL \
                --feature_levels $BIT_LEVEL \
                --QFeatureFlag True \
                --use_student_quant_params True \
                --use_adaptor True \
                --use_adaptor_bn False \
                --distill 'fd' \
                --distill_loss $DISTILL_LOSS \
                --model_type 'student' \
                --kd_gamma 1 \
                --kd_alpha $ALPHA \
                --epochs 1200 \
                --log_dir "./results/CIFAR10_ResNet20/${METHOD_TYPE}/alpha_${ALPHA}" \
                --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth'

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

total_time=$(( $end - $start ))
echo "RESULT: Total run time: $total_time seconds, started at $start_fmt"
