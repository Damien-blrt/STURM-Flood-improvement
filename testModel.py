import os
import shutil
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('./utils')

from utils.inference_metrics import run_inference
from utils.visualization import visualize_tile

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

base_dir = "./STURM-Flood/Dataset"

thresholds = [0.3,0.41,0.45,0.5,0.55]
#thresholds = np.arange(0.20,0.51,0.01)

SEED = 42

# ---------------------------------------------------------
# Save visualizations
# ---------------------------------------------------------

print("Do you want to save the visualisations of the predictions? (y/n)")
save_visuals = input().strip().lower() == 'y'

# ---------------------------------------------------------
# Available runs
# ---------------------------------------------------------

print("\nAvailable runs:")

for d in sorted(os.listdir("./save/runs")):
    print("\n")
    print(f"  - {d}")

    if os.path.exists(f"./save/runs/{d}/sentinel1/best_model.keras"):
        print("    - Sentinel-1 model found")

    if os.path.exists(f"./save/runs/{d}/sentinel2/best_model.keras"):
        print("    - Sentinel-2 model found")

# ---------------------------------------------------------
# Select run
# ---------------------------------------------------------

run_id = input(
    "\nEnter the run ID to evaluate: "
).strip()

while not os.path.exists(f"./save/runs/{run_id}"):

    print("Invalid run ID.")

    run_id = input(
        "Enter the run ID to evaluate: "
    ).strip()

# ---------------------------------------------------------
# Dataset choice
# ---------------------------------------------------------

sent_choice = input(
    "Enter dataset to evaluate "
    "(1 = Sentinel-1, 2 = Sentinel-2, 3 = both): "
).strip()

# ---------------------------------------------------------
# Dataset setup
# ---------------------------------------------------------

if sent_choice == '1':

    datasets_to_eval = ['Sentinel-1']

    s1_model_path = f"./save/runs/{run_id}/sentinel1/best_model.keras"

    sent1_dir = os.path.join(base_dir, "Sentinel1/S1")
    sent1_mask_dir = os.path.join(base_dir, "Sentinel1/Floodmaps")

    sent1_metadata = pd.read_csv(
        os.path.join(base_dir, "Sentinel1_metadata.csv")
    )

elif sent_choice == '2':

    datasets_to_eval = ['Sentinel-2']

    s2_model_path = f"./save/runs/{run_id}/sentinel2/best_model.keras"

    sent2_dir = os.path.join(base_dir, "Sentinel2/S2")
    sent2_mask_dir = os.path.join(base_dir, "Sentinel2/Floodmaps")

    sent2_metadata = pd.read_csv(
        os.path.join(base_dir, "Sentinel2_metadata.csv")
    )

else:

    datasets_to_eval = ['Sentinel-1', 'Sentinel-2']

    s1_model_path = f"./save/runs/{run_id}/sentinel1/best_model.keras"
    s2_model_path = f"./save/runs/{run_id}/sentinel2/best_model.keras"

    sent1_dir = os.path.join(base_dir, "Sentinel1/S1")
    sent2_dir = os.path.join(base_dir, "Sentinel2/S2")

    sent1_mask_dir = os.path.join(base_dir, "Sentinel1/Floodmaps")
    sent2_mask_dir = os.path.join(base_dir, "Sentinel2/Floodmaps")

    sent1_metadata = pd.read_csv(
        os.path.join(base_dir, "Sentinel1_metadata.csv")
    )

    sent2_metadata = pd.read_csv(
        os.path.join(base_dir, "Sentinel2_metadata.csv")
    )

# ---------------------------------------------------------
# Temporary predictions folder
# ---------------------------------------------------------

temp_output_dir = "./.tmp_preds"

os.makedirs(temp_output_dir, exist_ok=True)

# ---------------------------------------------------------
# FINAL VISUALIZATION FOLDER
# EXACT SAME OUTPUT AS SCRIPT 1
# ---------------------------------------------------------

visualization_dir = f"./save/runs/{run_id}/predictions_eval"

os.makedirs(visualization_dir, exist_ok=True)

# ---------------------------------------------------------
# Load models + recreate SAME test split
# ---------------------------------------------------------

if 'Sentinel-1' in datasets_to_eval:

    S1_SUBSET_SIZE = 5000

    sent1_subset = (
        sent1_metadata
        .sample(n=S1_SUBSET_SIZE, random_state=SEED)
        .reset_index(drop=True)
    )

    s1_train_size = int(S1_SUBSET_SIZE * 0.8)
    s1_val_size = int(S1_SUBSET_SIZE * 0.1)

    sent1_test = sent1_subset.iloc[s1_train_size + s1_val_size:]
    print("Sentinel-1 is loading (up to 5 minutes)")
    model_s1 = tf.keras.models.load_model(
        s1_model_path,
        compile=False
    )
    print(f"\nLoaded Sentinel-1 model")


if 'Sentinel-2' in datasets_to_eval:

    S2_SUBSET_SIZE = len(sent2_metadata)

    sent2_subset = (
        sent2_metadata
        .sample(n=S2_SUBSET_SIZE, random_state=SEED)
        .reset_index(drop=True)
    )

    s2_train_size = int(S2_SUBSET_SIZE * 0.8)
    s2_val_size = int(S2_SUBSET_SIZE * 0.1)

    sent2_test = sent2_subset.iloc[s2_train_size + s2_val_size:]
    print("Sentinel-2 is loading (up to 5 minutes)")
    model_s2 = tf.keras.models.load_model(
        s2_model_path,
        compile=False
    )
    print(f"\nLoaded Sentinel-2 model")

# ---------------------------------------------------------
# Composite / mask dirs
# ---------------------------------------------------------

composite_dirs = {}
mask_dirs = {}

if 'Sentinel-1' in datasets_to_eval:

    composite_dirs['Sentinel-1'] = sent1_dir
    mask_dirs['Sentinel-1'] = sent1_mask_dir

if 'Sentinel-2' in datasets_to_eval:

    composite_dirs['Sentinel-2'] = sent2_dir
    mask_dirs['Sentinel-2'] = sent2_mask_dir



# ---------------------------------------------------------
# Evaluation function
# ---------------------------------------------------------
def evaluate_model(
    val_df,
    dataset,
    model,
    is_s2=False,
    max_viz=10,
    max_eval=100
):

    print(f"\n--- Evaluating {dataset} on {max_eval} tiles ---")

    results = []

    for t in thresholds:

        print(f"\nThreshold = {t}")

        # -------------------------------------------------
        # VISUALIZATION FOLDER FOR THIS THRESHOLD
        # -------------------------------------------------

        threshold_visual_dir = os.path.join(
            visualization_dir,
            dataset,
            f"threshold_{t:.2f}"
        )

        

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        all_metrics = []

        for _, row in val_df.head(max_eval).iterrows():

            tile_id = row['tile_id']

            metrics = run_inference(
                tile_id,
                dataset,
                model,
                composite_dirs,
                mask_dirs,
                temp_output_dir,
                score_threshold=t,
                with_gt=True
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

        recall = tp / (tp + fn + 1e-7)

        f1_eau = (
            2 * precision * recall /
            (precision + recall + 1e-7)
        )

        iou_score = (
            tp / (tp + fp + fn + 1e-7)
        )

        accuracy = (
            (tp + tn) / total
        )

        support_eau = tp + fn
        support_non_eau = tn + fp

        p0 = tn / (tn + fn + 1e-7)

        r0 = tn / (tn + fp + 1e-7)

        f1_non_eau = (
            2 * p0 * r0 /
            (p0 + r0 + 1e-7)
        )

        weighted_f1_score = (
            f1_eau * support_eau +
            f1_non_eau * support_non_eau
        ) / total

        mean_metrics = {
            'Accuracy': accuracy,
            'Weighted F1': weighted_f1_score,
            'Dice Water': f1_eau,
            'Dice Non-Water': f1_non_eau,
            'Precision': precision,
            'Recall': recall,
            'IoU': iou_score,
            'Threshold': t,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn
        }

        print(f"\n{dataset} / Global Metrics pixel-wise:")

        for k, v in mean_metrics.items():

            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        results.append(mean_metrics)

        

        if save_visuals:
            os.makedirs(threshold_visual_dir, exist_ok=True)

            print(
                f"\nSaving {max_viz} visualizations "
                f"for threshold {t}..."
            )

            for _, row in val_df.head(max_viz).iterrows():

                tile_id = row['tile_id']

                # regenerate prediction
                run_inference(
                    tile_id,
                    dataset,
                    model,
                    composite_dirs,
                    mask_dirs,
                    temp_output_dir,
                    score_threshold=t,
                    with_gt=True
                )

                fig = visualize_tile(
                    tile_id,
                    dataset,
                    composite_dirs,
                    temp_output_dir,
                    mask_dirs,
                    is_s2=is_s2,
                    with_gt=True
                )

                fig.savefig(
                    os.path.join(
                        threshold_visual_dir,
                        f"{dataset}_{tile_id.split('.')[0]}.png"
                    ),
                    dpi=100,
                    bbox_inches='tight'
                )

                plt.close(fig)

    best = max(results, key=lambda x: x['Weighted F1'])

    print(f"\nBest threshold for {dataset}:")
    print(f"  Threshold: {best['Threshold']}")
    print(f"  Weighted F1: {best['Weighted F1']:.4f}")
    return results

# ---------------------------------------------------------
# Run evaluations and result graph
# ---------------------------------------------------------

if 'Sentinel-1' in datasets_to_eval:

    result_sent1 = evaluate_model(
        sent1_test,
        'Sentinel-1',
        model_s1,
        is_s2=False,
        max_viz=10,
        max_eval = len(sent1_test)
    )
    metrics_to_plot = [
        'Accuracy',
        'Weighted F1',
        'Dice Water',
        'Dice Non-Water',
        'Precision',
        'Recall',
        'IoU',
        'TP',
        'FP',
        'FN',
        'TN'
    ]
    df = pd.DataFrame(result_sent1)
    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(
            df['Threshold'],
            df[metric],
            marker='o'
        )
        plt.xlabel('Threshold')
        plt.ylabel(metric)
        plt.title(f'{metric} vs Threshold')
        plt.grid(True)
        plt.savefig(
            os.path.join(
                visualization_dir,
                f"{metric.replace(' ', '_').lower()}.png"
            )
        )
        plt.close()

if 'Sentinel-2' in datasets_to_eval:

    result_sent2 = evaluate_model(
        sent2_test,
        'Sentinel-2',
        model_s2,
        is_s2=True,
        max_eval=len(sent2_test),
        max_viz=10
    )
    metrics_to_plot = [
        'Accuracy',
        'Weighted F1',
        'Dice Water',
        'Dice Non-Water',
        'Precision',
        'Recall',
        'IoU',
        'TP',
        'FP',
        'FN',
        'TN'
    ]
    df = pd.DataFrame(result_sent2)
    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(
            df['Threshold'],
            df[metric],
            marker='o'
        )
        plt.xlabel('Threshold')
        plt.ylabel(metric)
        plt.title(f'{metric} vs Threshold')
        plt.grid(True)
        plt.savefig(
            os.path.join(
                visualization_dir,
                f"{metric.replace(' ', '_').lower()}.png"
            )
        )
        plt.close()


# ---------------------------------------------------------
# CLEAN ONLY TEMP PREDICTIONS
# ---------------------------------------------------------

print("\nCleaning temporary prediction files...")

if os.path.exists(temp_output_dir):

    shutil.rmtree(temp_output_dir)

    print(f"Removed: {temp_output_dir}")

# ---------------------------------------------------------
# End
# ---------------------------------------------------------

print("\nEvaluation completed.")

if save_visuals:

    print(f"\nVisualizations saved in:")
    print(visualization_dir)