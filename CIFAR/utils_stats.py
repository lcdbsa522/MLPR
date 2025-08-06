import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor


def feature_stats(fetures):
    num_channels = features.shape[1]
    feat_mean = torch.mean(features, dim=[0, 2, 3]).numpy()
    feat_std = torch.std(features, dim=[0, 2, 3]).numpy()
    feat_min = torch.min(features).item()
    feat_max = torch.max(features).item()
    feat_median = float(np.median(features.numpy().flatten()))

    feat_dict = {
        "feat_mean" : torch.mean(features, dim=[0, 2, 3]).numpy(),
        "feat_std" : torch.std(features, dim=[0, 2, 3]).numpy(),
        "feat_min" : torch.min(features).item(),
        "feat_max" : torch.max(features).item(),
        "feat_median" : float(np.median(features.numpy().flatten()))
    }

    return feat_dict


def count_unique_values(sample):
    """
    Args:
        sample (tensor) : 각 개별 이미지의 feature map (1, 64, 8, 8)
    """
    sample_flat = sample.flatten()
    unique_count = len(np.unique(sample_flat))
    return unique_count


def nunique_stats(features):
    num_samples = features.shape[0]
    sample_unique_counts = []

    features_np = features.numpy()

    print(f"\nCalculating unique values for {num_samples} samples...")

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(count_unique_values, features_np[i]) for i in range(num_samples)]

        for i, future in enumerate(tqdm(futures, desc="Processing samples")):
            sample_unique_counts.append(future.result())

    nunique_dict = {
        "unique_counts_mean" : np.mean(sample_unique_counts),
        "unique_counts_std" : np.std(sample_unique_counts),
        "unique_counts_min" : np.min(sample_unique_counts),
        "unique_counts_max" : np.max(sample_unique_counts),
        "unique_counts_median" : np.median(sample_unique_counts),
    }

    return nunique_dict, sample_unique_counts

def get_feature_quantizer_quant_level(nunique_dict, stats_method='median'):
    target_key = None
    for key in nunique_dict.keys():
        if stats_method in key:
            target_key = key
            break

    value = nunique_dict[target_key]

    n = math.floor(math.log2(value))
    threshold =  (2**n + 2**(n+1)) / 2

    quant_level = n if value <= threshold else n+1

    return 2**quant_level


def plot_unique_values_distribution(nunique_dict, sample_unique_counts, save_path=None):
    """
    Args:
        sample_unique_counts (list): 샘플별 고유값 수 리스트
        nunique_dict (dict): 고유값 통계 딕셔너리
        save_path (str, optional): 저장 경로. None이면 화면에 표시
    """
    plt.figure(figsize=(12, 6))

    # 히스토그램 + KDE
    plt.subplot(1, 2, 1)
    sns.histplot(sample_unique_counts, kde=True)
    plt.axvline(x=nunique_dict['unique_counts_mean'], color='r', linestyle='--', 
                label=f"Mean: {nunique_dict['unique_counts_mean']:.2f}")
    plt.axvline(x=nunique_dict['unique_counts_median'], color='g', linestyle='-.', 
                label=f"Median: {nunique_dict['unique_counts_median']:.2f}")
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
    
    if save_path:
        # 디렉토리 확인 및 생성
        save_dir = os.path.dirname(os.path.join(save_path, 'feature_analysis'))
        os.makedirs(save_dir, exist_ok=True)
        
        # 파일 저장
        plt.savefig(os.path.join(save_dir, 'unique_values_distribution.png'))
        plt.close()
    else:
        plt.show()


def plot_feature_maps(features, save_path=None):
    """
    Args:
        features (torch.Tensor): 특징 맵 텐서 [N, C, H, W]
        save_path (str, optional): 저장 경로. None이면 화면에 표시
    """
    num_channels = features.shape[1]
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
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def save_feature_stats(stats, save_path):
    """    
    Args:
        stats (dict): 저장할 통계 정보
        save_path (str): 저장 경로
    """
    os.makedirs(save_path, exist_ok=True)

    converted_stats = convert_numpy_types(stats)
    
    json_path = os.path.join(save_path, 'feature_stats.json')
    with open(json_path, 'w') as f:
        json.dump(converted_stats, f, indent=4)