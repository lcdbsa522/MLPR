import pandas as pd
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from scipy import stats as scipy_stats
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor

from dataset.cifar10 import get_cifar10_dataloaders
from dataset.cifar100 import get_cifar100_dataloaders
from models.custom_models_resnet import *
from utils import *

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description="Feature Map Analysis for Quantized Models")
parser.add_argument('--dataset', type=str, default='cifar10', choices=('cifar10','cifar100'), help='사용할 데이터셋 CIFAR10|CIFAR100')
parser.add_argument('--model_type', type=str, default='teacher', choices=['teacher','student'], help='모델 타입: teacher 또는 student')
parser.add_argument('--arch', type=str, default=None, help='모델 아키텍처; 지정하지 않으면 model_type에 따라 기본값 사용 (teacher: resnet20_fp, student: resnet20_quant)')
parser.add_argument('--batch_size', type=int, default=256, help='특징 추출 배치 사이즈')
parser.add_argument('--num_workers', type=int, default=4, help='데이터 로딩 워커 수')
parser.add_argument('--split', type=str, default='test', choices=('train', 'test'), help='사용할 데이터셋 분할 (train 또는 test)')
parser.add_argument('--gpu_id', type=str, default='0', help='사용할 GPU ID')
parser.add_argument('--seed', type=int, default=None, help='seed for initialization')

# 체크포인트 경로
parser.add_argument('--checkpoint_path', type=str, default=None, help='모델 체크포인트 경로; 지정하지 않으면 model_type에 따라 기본값 사용')

# 분석 결과 저장 디렉토리
parser.add_argument('--log_dir', type=str, default='./feature_analysis', help='분석 결과 저장 디렉토리')

# 양자화 관련 인자들
parser.add_argument('--QWeightFlag', type=str2bool, default=True, help='가중치 양자화 여부')
parser.add_argument('--QActFlag', type=str2bool, default=True, help='활성화 양자화 여부')
parser.add_argument('--QFeatureFlag_t', type=str2bool, default=False, help='teacher feature 양자화 여부')
parser.add_argument('--QFeatureFlag_s', type=str2bool, default=False, help='student feature 양자화 여부')
parser.add_argument('--weight_levels', type=int, default=2, help='가중치 양자화 레벨 수')
parser.add_argument('--act_levels', type=int, default=2, help='활성화 양자화 레벨 수')
parser.add_argument('--feature_levels', type=int, default=2, help='feature 양자화 레벨 수')
parser.add_argument('--use_student_quant_params', type=str2bool, default=True, help='teacher 양자화 시 student 양자화 파라미터 사용 여부')
parser.add_argument('--use_adapter', type=str2bool, default=False, help='student 모델에 adapter 사용 여부')
parser.add_argument('--use_binary_heatmap', type=str2bool, default=False, help='이진 히트맵 사용 여부')
parser.add_argument('--weight_type', type=str, default=None, choices=['cam','grad_cam','gap_grad_cam'], help='Feature Map Weight Method')
parser.add_argument('--bkwd_scaling_factorF', type=float, default=0.0, help='feature 양자화를 위한 스케일링 팩터')

parser.add_argument('--baseline', type=str2bool, default=False, help='training with STE')
parser.add_argument('--bkwd_scaling_factorW', type=float, default=0.0, help='scaling factor for weights')
parser.add_argument('--bkwd_scaling_factorA', type=float, default=0.0, help='scaling factor for activations')
parser.add_argument('--use_hessian', type=str2bool, default=True, help='update scaling factor using Hessian trace')
parser.add_argument('--update_every', type=int, default=10, help='update interval in terms of epochs')
parser.add_argument('--quan_method', type=str, default='EWGS', help='training with different quantization methods')

args = parser.parse_args()

def _init_fn(worker_id):
    seed = args.seed + worker_id
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

# 데이터셋에 따른 클래스 수 설정
if args.dataset == 'cifar10':
    args.num_classes = 10
    train_dataset, test_dataset = get_cifar10_dataloaders(data_folder="./dataset/data/CIFAR10/")

elif args.dataset == 'cifar100':
    args.num_classes = 100
    train_dataset, test_dataset = get_cifar100_dataloaders(data_folder="./dataset/data/CIFAR100/")

if args.split == 'train':
    data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=None if args.seed is None else _init_fn
    )
elif args.split == 'test':
    data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=args.num_workers
    )


# 모델 아키텍처 기본값 설정
if args.arch is None:
    if args.model_type == 'teacher':
        args.arch = 'resnet20_fp'
    elif args.model_type == 'student':
        args.arch = 'resnet20_quant'
    else:
        raise ValueError("Invalid model_type.")

# 체크포인트 경로 기본값 설정
if args.checkpoint_path is None:
    if args.model_type == 'teacher':
        args.checkpoint_path = './results/CIFAR10_ResNet20/fp/checkpoint/last_checkpoint.pth'
    elif args.model_type == 'student':
        args.checkpoint_path = './results/CIFAR10_ResNet20/FQA(s)_4bit/checkpoint/last_checkpoint.pth'
    else:
        raise ValueError("Invalid model_type.")

# GPU 설정 및 결과 저장 디렉토리 생성
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(args.log_dir, exist_ok=True)


# 모델 로딩 (동적 호출)
model_class = globals().get(args.arch)
model = model_class(args)
model.to(device)

# 체크포인트 로드 및 모델 weight 초기화 (teacher와 student 모두 동일한 방식)
checkpoint = torch.load(args.checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])
print(f"{args.model_type.capitalize()} model loaded from {args.checkpoint_path}, test accuracy: {checkpoint['test_acc']:.2f}%")

# 특징 추출 (전체 데이터셋 사용)
model.eval()
features_list = []
labels_list = []
with torch.no_grad():
    for images, labels in tqdm(data_loader, desc="Extracting features"):
        images = images.to(device)
        features = model.feature_extractor(images)
        features_list.append(features.detach().cpu())
        labels_list.append(labels)
features = torch.cat(features_list, dim=0)
labels = torch.cat(labels_list, dim=0)

# 특징 분석 및 시각화
num_channels = features.shape[1]
feat_mean = torch.mean(features, dim=[0, 2, 3]).numpy()
feat_std = torch.std(features, dim=[0, 2, 3]).numpy()
feat_min = torch.min(features).item()
feat_max = torch.max(features).item()
feat_median = float(np.median(features.numpy().flatten()))

# 각 샘플별 고유값 수 계산
def count_unique_values(sample):
    """단일 샘플의 고유값 수를 계산하는 함수"""
    # 샘플을 1차원 배열로 변환
    sample_flat = sample.flatten()
    # 고유값 수 계산
    unique_count = len(np.unique(sample_flat))
    return unique_count

# 샘플별 고유값 수 계산 (병렬 처리)
num_samples = features.shape[0]
sample_unique_counts = []

# 병렬 처리를 위해 각 샘플을 numpy 배열로 변환
features_np = features.numpy()

print(f"\nCalculating unique values for {num_samples} samples...")
# ProcessPoolExecutor를 사용한 병렬 처리
with ProcessPoolExecutor() as executor:
    # 각 샘플에 대해 고유값 수 계산 작업 제출
    futures = [executor.submit(count_unique_values, features_np[i]) for i in range(num_samples)]
    
    # 결과 수집 (진행 상황 표시와 함께)
    for i, future in enumerate(tqdm(futures, desc="Processing samples")):
        sample_unique_counts.append(future.result())

# 샘플별 고유값 수에 대한 통계
unique_counts_mean = np.mean(sample_unique_counts)
unique_counts_std = np.std(sample_unique_counts)
unique_counts_min = np.min(sample_unique_counts)
unique_counts_max = np.max(sample_unique_counts)
unique_counts_median = np.median(sample_unique_counts)

print("\nUnique Value Counts per Sample Statistics:")
print(f"Mean: {unique_counts_mean:.2f}")
print(f"Std: {unique_counts_std:.2f}")
print(f"Min: {unique_counts_min}")
print(f"Max: {unique_counts_max}")
print(f"Median: {unique_counts_median}")

# 샘플별 고유값 수 분포 시각화
plt.figure(figsize=(12, 6))

# 히스토그램 + KDE
plt.subplot(1, 2, 1)
sns.histplot(sample_unique_counts, kde=True)
plt.axvline(x=unique_counts_mean, color='r', linestyle='--', label=f'Mean: {unique_counts_mean:.2f}')
plt.axvline(x=unique_counts_median, color='g', linestyle='-.', label=f'Median: {unique_counts_median:.2f}')
plt.title('Distribution of Unique Value Counts per Sample')
plt.xlabel('Number of Unique Values')
plt.ylabel('Frequency')
plt.legend()

# 박스 플롯
plt.subplot(1, 2, 2)
sns.boxplot(y=sample_unique_counts)
plt.title('Boxplot of Unique Value Counts per Sample')
plt.ylabel('Number of Unique Values')

plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, 'sample_unique_values_distribution.png'))
plt.close()

# 클래스별로 고유값 수 분포 비교
class_unique_counts = {}
for c in range(args.num_classes):
    class_mask = (labels == c).numpy()
    if np.sum(class_mask) > 0:
        class_unique_counts[c] = [sample_unique_counts[i] for i, is_class in enumerate(class_mask) if is_class]

# 클래스별 분포 시각화 (최대 10개 클래스만)
classes_to_plot = sorted(class_unique_counts.keys())[:10]
if len(classes_to_plot) > 1:  # 클래스가 2개 이상일 때만 시각화
    plt.figure(figsize=(14, 6))
    
    # 클래스별 박스플롯
    plt.subplot(1, 2, 1)
    box_data = [class_unique_counts[c] for c in classes_to_plot]
    plt.boxplot(box_data, labels=[f'Class {c}' for c in classes_to_plot])
    plt.title('Unique Value Counts by Class')
    plt.ylabel('Number of Unique Values')
    plt.grid(alpha=0.3)
    
    # 클래스별 바이올린 플롯
    plt.subplot(1, 2, 2)
    violin_data = []
    violin_labels = []
    for c in classes_to_plot:
        violin_data.extend(class_unique_counts[c])
        violin_labels.extend([f'Class {c}'] * len(class_unique_counts[c]))
    
    violin_df = pd.DataFrame({'Class': violin_labels, 'Unique Count': violin_data})
    sns.violinplot(x='Class', y='Unique Count', data=violin_df)
    plt.title('Violin Plot of Unique Value Counts by Class')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, 'class_unique_values_distribution.png'))
    plt.close()

# 고유값 수와 label 간의 상관관계 확인
if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    plt.figure(figsize=(10, 6))
    scatter_df = pd.DataFrame({'Label': labels.numpy(), 'Unique Count': sample_unique_counts})
    sns.stripplot(x='Label', y='Unique Count', data=scatter_df, alpha=0.5, jitter=True)
    plt.title('Unique Value Counts vs Class Label')
    plt.xlabel('Class Label')
    plt.ylabel('Number of Unique Values')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, 'label_vs_unique_values.png'))
    plt.close()

# 통계 정보 JSON에 추가
sample_stats = {
    'mean': float(unique_counts_mean),
    'std': float(unique_counts_std),
    'min': int(unique_counts_min),
    'max': int(unique_counts_max),
    'median': float(unique_counts_median),
    'all_counts': sample_unique_counts  # 모든 샘플의 고유값 수 리스트
}

# 클래스별 통계 추가
class_stats = {}
for c in class_unique_counts:
    if len(class_unique_counts[c]) > 0:
        class_stats[str(c)] = {
            'mean': float(np.mean(class_unique_counts[c])),
            'std': float(np.std(class_unique_counts[c])),
            'min': int(np.min(class_unique_counts[c])),
            'max': int(np.max(class_unique_counts[c])),
            'median': float(np.median(class_unique_counts[c])),
            'count': len(class_unique_counts[c])
        }

# 채널별 분산 계산
channel_variances = torch.var(features, dim=[0, 2, 3]).numpy()

# 채널별 Zero 값 비율 계산
zero_counts = torch.sum(features == 0.0, dim=[0, 2, 3]).float()
total_elements_per_channel = float(features.shape[0] * features.shape[2] * features.shape[3])
zero_ratios = (zero_counts / total_elements_per_channel).numpy()

# 전체 Zero 값 비율
total_zero_ratio = torch.sum(features == 0.0).item() / features.numel()

# 각 채널의 스파스성 (0이 아닌 값 대비 0인 값의 비율)
sparsity = torch.mean((features == 0.0).float()).item()

# 정적인 히스토그램 계산
hist_values, hist_bins = np.histogram(features.numpy().flatten(), bins=100)

# 양/음수 값 비율
positive_ratio = torch.mean((features > 0).float()).item()
negative_ratio = torch.mean((features < 0).float()).item()
zero_ratio = torch.mean((features == 0).float()).item()

print("\nFeature Statistics:")
print(f"Mean: {np.mean(feat_mean):.4f}, Std: {np.mean(feat_std):.4f}, Min: {feat_min:.4f}, Max: {feat_max:.4f}")
print(f"Median: {feat_median:.4f}, Q1: {feat_q1:.4f}, Q3: {feat_q3:.4f}, IQR: {feat_iqr:.4f}")
print(f"Skewness: {feat_skewness:.4f}, Kurtosis: {feat_kurtosis:.4f}")
print(f"Sparsity (Zero ratio): {sparsity:.4f}")
print(f"Positive values ratio: {positive_ratio:.4f}, Negative values ratio: {negative_ratio:.4f}, Zero ratio: {zero_ratio:.4f}")

# 시각화 1: 특징 값 분포
plt.figure(figsize=(14, 10))

# 1-1: 히스토그램
plt.subplot(2, 2, 1)
plt.hist(features.numpy().flatten(), bins=100, alpha=0.7)
plt.title('Histogram of Feature Values')
plt.xlabel('Feature Value')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)

# 1-2: KDE 플롯
plt.subplot(2, 2, 2)
sns.kdeplot(features.numpy().flatten(), fill=True)
plt.title('KDE of Feature Values')
plt.xlabel('Feature Value')
plt.ylabel('Density')
plt.grid(alpha=0.3)

# 1-3: 누적 분포 함수
plt.subplot(2, 2, 3)
plt.hist(features.numpy().flatten(), bins=100, cumulative=True, density=True, alpha=0.7)
plt.title('Cumulative Distribution of Feature Values')
plt.xlabel('Feature Value')
plt.ylabel('Cumulative Probability')
plt.grid(alpha=0.3)

# 1-4: 박스 플롯 (채널별)
plt.subplot(2, 2, 4)
channel_data = [features[:, c, :, :].flatten().numpy() for c in range(min(num_channels, 10))]
plt.boxplot(channel_data)
plt.title('Boxplot of Feature Values (First 10 Channels)')
plt.xlabel('Channel Index')
plt.ylabel('Feature Value')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, 'feature_distributions.png'))
plt.close()

# 시각화 2: 채널별 통계
plt.figure(figsize=(14, 10))

# 2-1: 채널별 평균값
plt.subplot(2, 2, 1)
plt.bar(range(num_channels), feat_mean, alpha=0.7)
plt.title('Channel-wise Mean Values')
plt.xlabel('Channel Index')
plt.ylabel('Mean Value')
plt.grid(alpha=0.3)

# 2-2: 채널별 표준편차
plt.subplot(2, 2, 2)
plt.bar(range(num_channels), feat_std, alpha=0.7, color='orange')
plt.title('Channel-wise Standard Deviation')
plt.xlabel('Channel Index')
plt.ylabel('Std Value')
plt.grid(alpha=0.3)

# 2-3: 채널별 Zero 값 비율
plt.subplot(2, 2, 3)
plt.bar(range(num_channels), zero_ratios, alpha=0.7, color='green')
plt.title('Channel-wise Zero Ratio')
plt.xlabel('Channel Index')
plt.ylabel('Zero Ratio')
plt.grid(alpha=0.3)

# 2-4: 채널별 분산
plt.subplot(2, 2, 4)
plt.bar(range(num_channels), channel_variances, alpha=0.7, color='red')
plt.title('Channel-wise Variance')
plt.xlabel('Channel Index')
plt.ylabel('Variance')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, 'channel_statistics.png'))
plt.close()

# 시각화 3: Feature Map 히트맵 (첫 몇 샘플, 몇 개 채널)
num_samples_viz = min(5, features.shape[0])
num_channels_viz = min(4, num_channels)
plt.figure(figsize=(15, 10))
for i in range(num_samples_viz):
    for j in range(num_channels_viz):
        plt.subplot(num_samples_viz, num_channels_viz, i * num_channels_viz + j + 1)
        plt.imshow(features[i, j].numpy(), cmap='viridis')
        plt.title(f'Sample {i}, Ch {j}')
        plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, 'feature_maps.png'))
plt.close()

# 시각화 4: 클래스별 특징 분석 (각 클래스별 채널 평균)
class_means = {}
class_stds = {}
class_zero_ratios = {}
for c in range(args.num_classes):
    class_mask = (labels == c)
    if torch.sum(class_mask) > 0:
        class_features = features[class_mask]
        class_means[c] = torch.mean(class_features, dim=[0, 2, 3]).numpy()
        class_stds[c] = torch.std(class_features, dim=[0, 2, 3]).numpy()
        
        # 클래스별 Zero 값 비율 계산
        class_zero_counts = torch.sum(class_features == 0.0, dim=[0, 2, 3]).float()
        class_total_elements = float(class_features.shape[0] * class_features.shape[2] * class_features.shape[3])
        class_zero_ratios[c] = (class_zero_counts / class_total_elements).numpy()

plt.figure(figsize=(15, 15))
# 4-1: 클래스별 채널 평균
for i, c in enumerate(sorted(class_means.keys())[:6]):
    plt.subplot(3, 6, i + 1)
    plt.bar(range(num_channels), class_means[c], alpha=0.7)
    plt.title(f'Class {c} Channel Means')
    plt.xlabel('Channel')
    plt.ylabel('Mean')
    plt.grid(alpha=0.3)

# 4-2: 클래스별 채널 표준편차
for i, c in enumerate(sorted(class_stds.keys())[:6]):
    plt.subplot(3, 6, i + 7)
    plt.bar(range(num_channels), class_stds[c], alpha=0.7, color='orange')
    plt.title(f'Class {c} Channel Stds')
    plt.xlabel('Channel')
    plt.ylabel('Std')
    plt.grid(alpha=0.3)

# 4-3: 클래스별 채널 Zero 비율
for i, c in enumerate(sorted(class_zero_ratios.keys())[:6]):
    plt.subplot(3, 6, i + 13)
    plt.bar(range(num_channels), class_zero_ratios[c], alpha=0.7, color='green')
    plt.title(f'Class {c} Zero Ratios')
    plt.xlabel('Channel')
    plt.ylabel('Zero Ratio')
    plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, 'class_feature_statistics.png'))
plt.close()

# 시각화 5: 평균 공간 활성화 히트맵
plt.figure(figsize=(15, 5))

# 5-1: 채널 축 평균 (전체)
plt.subplot(1, 3, 1)
spatial_mean = torch.mean(features, dim=1)  # 채널 축 평균
plt.imshow(spatial_mean.mean(dim=0).numpy(), cmap='hot')
plt.title('Average Spatial Activation (All)')
plt.colorbar()

# 5-2: 채널 축 평균 (양수 값만)
plt.subplot(1, 3, 2)
positive_mask = features > 0
positive_features = features * positive_mask.float()
spatial_pos_mean = torch.mean(positive_features, dim=1)  # 채널 축 평균
plt.imshow(spatial_pos_mean.mean(dim=0).numpy(), cmap='hot')
plt.title('Average Spatial Activation (Positive)')
plt.colorbar()

# 5-3: 채널 축 평균 (음수 값만)
plt.subplot(1, 3, 3)
negative_mask = features < 0
negative_features = features * negative_mask.float()
spatial_neg_mean = torch.mean(negative_features, dim=1)  # 채널 축 평균
plt.imshow(spatial_neg_mean.mean(dim=0).numpy(), cmap='hot')
plt.title('Average Spatial Activation (Negative)')
plt.colorbar()

plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, 'spatial_activation.png'))
plt.close()

# 시각화 6: 고유값 분석
plt.figure(figsize=(12, 5))

# 각 샘플별 정밀도에 따른 고유값 수 변화 분석
# 전체 데이터셋 중 일부 샘플만 선택하여 분석 (최대 10개)
sample_indices = np.random.choice(num_samples, min(10, num_samples), replace=False)
selected_samples = [features_np[i].flatten() for i in sample_indices]

# 정밀도에 따른 고유값 수 변화 분석을 위한 정밀도 범위
precisions = [0, 1, 2, 3, 4, 5, 6]  # 반올림 정밀도 범위
precision_unique_counts = []

for sample_idx, sample_data in enumerate(selected_samples):
    sample_unique_counts_by_precision = []
    for p in precisions:
        rounded = np.round(sample_data, p)
        sample_unique_counts_by_precision.append(len(np.unique(rounded)))
    precision_unique_counts.append(sample_unique_counts_by_precision)

# 6-1: 정밀도별 고유값 수
plt.subplot(1, 2, 1)
for i, counts in enumerate(precision_unique_counts):
    plt.plot(precisions, counts, 'o-', linewidth=1, label=f'Sample {i+1}')
plt.title('Unique Values Count by Precision')
plt.xlabel('Decimal Precision')
plt.ylabel('Unique Values Count')
plt.grid(alpha=0.3)
plt.xticks(precisions)
if len(precision_unique_counts) <= 5:  # 샘플이 많지 않을 때만 범례 표시
    plt.legend()

# 6-2: 정밀도별 고유값 수 (로그 스케일)
plt.subplot(1, 2, 2)
for i, counts in enumerate(precision_unique_counts):
    plt.plot(precisions, counts, 'o-', linewidth=1, label=f'Sample {i+1}')
plt.yscale('log')
plt.title('Unique Values Count (Log Scale)')
plt.xlabel('Decimal Precision')
plt.ylabel('Unique Values Count (Log)')
plt.grid(alpha=0.3)
plt.xticks(precisions)
if len(precision_unique_counts) <= 5:  # 샘플이 많지 않을 때만 범례 표시
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(args.log_dir, 'unique_values_analysis.png'))
plt.close()

# 통계 정보를 JSON 파일로 저장
stats = {
    'basic_stats': {
        'mean': float(np.mean(feat_mean)),
        'std': float(np.mean(feat_std)),
        'min': feat_min,
        'max': feat_max,
        'median': feat_median,
        'q1': feat_q1,
        'q3': feat_q3,
        'iqr': feat_iqr
    },
    'distribution_stats': {
        'skewness': float(feat_skewness),
        'kurtosis': float(feat_kurtosis),
        'sparsity': sparsity,
        'positive_ratio': positive_ratio,
        'negative_ratio': negative_ratio,
        'zero_ratio': zero_ratio
    },
    'channel_stats': {
        'channel_means': feat_mean.tolist(),
        'channel_stds': feat_std.tolist(),
        'channel_zero_ratios': zero_ratios.tolist(),
        'channel_variances': channel_variances.tolist()
    },
    'sample_unique_values': sample_stats,
    'class_unique_values': class_stats,
    'histogram': {
        'bins': hist_bins.tolist(),
        'values': hist_values.tolist()
    }
}

with open(os.path.join(args.log_dir, 'feature_stats.json'), 'w') as f:
    json.dump(stats, f, indent=4)

print(f"\nEnhanced analysis results saved to {args.log_dir}")