import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('./arch')
sys.path.append('./utils')

from model import unet_model
from utils.inference_metrics import run_inference

# ---------------------------------------------------------
# Pretrained models
# ---------------------------------------------------------

def rebuild_model(params):
    model = unet_model(
        n_classes=2,
        tile_width=params['input_shape'][0],
        tile_height=params['input_shape'][1],
        n_bands=params['input_shape'][2],
        n_blocks=params['n_blocks'],
        class_weight_list=[1, 1],
        normalize_inputs=False
    )
    model.load_weights(params['weights_path'])
    return model

s1_model = rebuild_model({
    'input_shape': (128, 128, 2),
    'n_blocks': 6,
    'weights_path': './unet/sentinel1_model/unet/1/model_weights.hdf5'
})

s2_model = rebuild_model({
    'input_shape': (128, 128, 9),
    'n_blocks': 5,
    'weights_path': './unet/sentinel2_model/unet/1/model_weights.hdf5'
})

# ---------------------------------------------------------
# Data paths and test sample selection
# ---------------------------------------------------------

composite_dirs = {
    'Sentinel-1': './STURM-Flood/Dataset/Sentinel1/S1',
    'Sentinel-2': './STURM-Flood/Dataset/Sentinel2/S2'
}
mask_dirs = {
    'Sentinel-1': './STURM-Flood/Dataset/Sentinel1/Floodmaps',
    'Sentinel-2': './STURM-Flood/Dataset/Sentinel2/Floodmaps'
}
output_dir = './tmp_inference_results'
os.makedirs(output_dir, exist_ok=True)

s1_metadata = pd.read_csv('./STURM-Flood/Dataset/Sentinel1_metadata.csv')
s2_metadata = pd.read_csv('./STURM-Flood/Dataset/Sentinel2_metadata.csv')

s1_test_df = s1_metadata.sample(n=500, random_state=42)
s2_test_df = s2_metadata.sample(n=200, random_state=42)

# ---------------------------------------------------------
# Run inference and calculate metrics for test samples
# ---------------------------------------------------------

results = []

for dataset, df, model in zip(
    ['Sentinel-1', 'Sentinel-2'],
    [s1_test_df, s2_test_df],
    [s1_model, s2_model]
):
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset}"):
        metrics = run_inference(row['tile_id'], dataset, model, composite_dirs, mask_dirs, output_dir, with_gt=True)
        if metrics:
            metrics['Dataset'] = dataset
            results.append(metrics)

# ---------------------------------------------------------
#  Aggregate and display results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n=== GLOBAL PIXEL-WISE METRICS (comme le papier) ===")
for dataset in ['Sentinel-1', 'Sentinel-2']:
    df = results_df[results_df['Dataset'] == dataset]
    tp = df['TP'].sum()
    fp = df['FP'].sum()
    fn = df['FN'].sum()
    tn = df['TN'].sum()
    total = tp + fp + fn + tn

    
    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1_eau    = 2 * precision * recall / (precision + recall + 1e-7)
    iou       = tp / (tp + fp + fn + 1e-7)
    accuracy  = (tp + tn) / total


    support_eau     = tp + fn
    support_non_eau = tn + fp
    p0 = tn / (tn + fn + 1e-7)
    r0 = tn / (tn + fp + 1e-7)
    f1_non_eau  = 2 * p0 * r0 / (p0 + r0 + 1e-7)
    weighted_f1 = (f1_eau * support_eau + f1_non_eau * support_non_eau) / total

    print(f"\n{dataset}:")
    print(f"  Accuracy     : {accuracy:.4f}")
    print(f"  Weighted F1  : {weighted_f1:.4f}")
    print(f"  F1 eau       : {f1_eau:.4f}")
    print(f"  Precision    : {precision:.4f}")
    print(f"  Recall       : {recall:.4f}")
    print(f"  IoU          : {iou:.4f}")