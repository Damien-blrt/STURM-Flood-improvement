import os
import shutil
import pandas as pd
import tensorflow as tf
import sys

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
run_id = "20260507_031410" # Update this to match your actual run ID
base_dir = "./STURM-Flood/Dataset"
threshold = [0.525, 0.7, 0.3, 0.5] 

s1_model_path = f"./save/runs/{run_id}/sentinel1/best_model.keras"
s2_model_path = f"./save/runs/{run_id}/sentinel2/best_model.keras"

sent1_dir = os.path.join(base_dir, "Sentinel1/S1")
sent2_dir = os.path.join(base_dir, "Sentinel2/S2")
sent1_mask_dir = os.path.join(base_dir, "Sentinel1/Floodmaps")
sent2_mask_dir = os.path.join(base_dir, "Sentinel2/Floodmaps")

# ---------------------------------------------------------
# Temp output (fix crash)
# ---------------------------------------------------------
temp_output_dir = "./tmp_preds"
os.makedirs(temp_output_dir, exist_ok=True)

# ---------------------------------------------------------
# Load metadata + same split
# ---------------------------------------------------------
sent1_metadata = pd.read_csv(os.path.join(base_dir, "Sentinel1_metadata.csv"))
sent2_metadata = pd.read_csv(os.path.join(base_dir, "Sentinel2_metadata.csv"))

SEED = 1
S1_SUBSET_SIZE = 5000
S2_SUBSET_SIZE = len(sent2_metadata)

sent1_subset = sent1_metadata.sample(n=S1_SUBSET_SIZE, random_state=SEED).reset_index(drop=True)
sent2_subset = sent2_metadata.sample(n=S2_SUBSET_SIZE, random_state=SEED).reset_index(drop=True)

s1_train_size = int(S1_SUBSET_SIZE * 0.8)
s1_val_size   = int(S1_SUBSET_SIZE * 0.1)

s2_train_size = int(S2_SUBSET_SIZE * 0.8)
s2_val_size   = int(S2_SUBSET_SIZE * 0.1)

sent1_test = sent1_subset.iloc[s1_train_size + s1_val_size:]
sent2_test = sent2_subset.iloc[s2_train_size + s2_val_size:]

# ---------------------------------------------------------
# Load models (no compile)
# ---------------------------------------------------------
model_s1 = tf.keras.models.load_model(s1_model_path, compile=False)
model_s2 = tf.keras.models.load_model(s2_model_path, compile=False)

print("Models loaded successfully")

# ---------------------------------------------------------
# Import inference
# ---------------------------------------------------------
sys.path.append('./utils')
from utils.inference_metrics import run_inference

composite_dirs = {
    'Sentinel-1': sent1_dir,
    'Sentinel-2': sent2_dir
}
mask_dirs = {
    'Sentinel-1': sent1_mask_dir,
    'Sentinel-2': sent2_mask_dir
}

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
def evaluate_model(val_df, dataset, model, max_eval=200):
    print(f"\n--- Evaluating {dataset} on {max_eval} tiles ---")
    results = []
    for t in threshold: 
        all_metrics = []

        for _, row in val_df.head(max_eval).iterrows():
            tile_id = row['tile_id']

            metrics = run_inference(
                tile_id,
                dataset,
                model,
                composite_dirs,
                mask_dirs,
                output_dir=temp_output_dir,
                with_gt=True,
                score_threshold=t
            )

            if metrics:
                all_metrics.append(metrics)

        df = pd.DataFrame(all_metrics)

        tp = df['TP'].sum()
        fp = df['FP'].sum()
        fn = df['FN'].sum()
        tn = df['TN'].sum()
        total = tp + fp + fn + tn

        precision = tp / (tp + fp + 1e-7)
        recall    = tp / (tp + fn + 1e-7)
        f1_water        = 2 * precision * recall / (precision + recall + 1e-7)
        f1_non_water    = 2 * (tn / (tn + fn + 1e-7)) * (tn / (tn + fp + 1e-7)) / ((tn / (tn + fn + 1e-7)) + (tn / (tn + fp + 1e-7)) + 1e-7)
        weighted_f1     = (f1_water * (tp + fn) + f1_non_water * (tn + fp)) / total
        iou       = tp / (tp + fp + fn + 1e-7)
        accuracy  = (tp + tn) / total
        results.append({
            'Dataset': dataset,
            'Threshold': t,
            'F1': weighted_f1
        })

        print(f"\n{dataset} results {t}:")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  F1 Water      : {f1_water:.4f}")
        print(f"  F1 Non-Water  : {f1_non_water:.4f}")
        print(f"  Weighted F1   : {weighted_f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  IoU      : {iou:.4f}")
    print(f"best threshold for {dataset}: {max(results, key=lambda x: x['F1'])['Threshold']}")


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------
evaluate_model(sent1_test, 'Sentinel-1', model_s1)
evaluate_model(sent2_test, 'Sentinel-2', model_s2)

# ---------------------------------------------------------
# Optional cleanup
# ---------------------------------------------------------
os.system(f'rm -rf {temp_output_dir}')
print("\nTemporary files cleaned.")