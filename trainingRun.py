import os
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
import sys
from utils.utility import preprocess_mask, write_geotiff
from utils.inference_metrics import run_inference
from utils.visualization import visualize_tile, save_visualizations
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

sys.path.append('./arch')
sys.path.append('./utils')
from model import unet_model

# ---------------------------------------------------------
# Folders creation
# ---------------------------------------------------------

os.system('rm -rf logs')

#os.system('rm -rf ./logs/sent1/*')
#os.system('rm -rf ./logs/sent2/*')

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
s1_save_dir = f"./save/runs/{run_id}/sentinel1"
s2_save_dir = f"./save/runs/{run_id}/sentinel2"
os.makedirs(s1_save_dir, exist_ok=True)
os.makedirs(s2_save_dir, exist_ok=True)
logs_dir = "./logs"
sent1_log_dir = "./logs/sent1"
sent2_log_dir = "./logs/sent2"
base_dir = "./STURM-Flood/Dataset"
sent1_dir = os.path.join(base_dir, "Sentinel1/S1")
sent2_dir = os.path.join(base_dir, "Sentinel2/S2")
sent1_mask_dir = os.path.join(base_dir, "Sentinel1/Floodmaps")
sent2_mask_dir = os.path.join(base_dir, "Sentinel2/Floodmaps")
sent1_metadata_path = os.path.join(base_dir, "Sentinel1_metadata.csv")
sent2_metadata_path = os.path.join(base_dir, "Sentinel2_metadata.csv")
s1_model_dir = './unet/sentinel1_model'
s2_model_dir = './unet/sentinel2_model'
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(sent1_log_dir, exist_ok=True)
os.makedirs(sent2_log_dir, exist_ok=True)
os.makedirs("cache", exist_ok=True)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

sent1_metadata = pd.read_csv(sent1_metadata_path)
sent2_metadata = pd.read_csv(sent2_metadata_path)

SEED = 42
BATCH_SIZE = 16

S1_SUBSET_SIZE = 5000 #len(sent1_metadata)
S2_SUBSET_SIZE = len(sent2_metadata)

s1_train_size = int(S1_SUBSET_SIZE * 0.8)
s1_val_size   = int(S1_SUBSET_SIZE * 0.1)

s2_train_size = int(S2_SUBSET_SIZE * 0.8)
s2_val_size   = int(S2_SUBSET_SIZE * 0.1)

sent1_subset = sent1_metadata.sample(n=S1_SUBSET_SIZE, random_state=SEED).reset_index(drop=True)
sent2_subset = sent2_metadata.sample(n=S2_SUBSET_SIZE, random_state=SEED).reset_index(drop=True)


sent1_train = sent1_subset.iloc[:s1_train_size]
sent1_val   = sent1_subset.iloc[s1_train_size:s1_train_size + s1_val_size]
sent1_test  = sent1_subset.iloc[s1_train_size + s1_val_size:]

sent2_train = sent2_subset.iloc[:s2_train_size]
sent2_val   = sent2_subset.iloc[s2_train_size:s2_train_size + s2_val_size]
sent2_test  = sent2_subset.iloc[s2_train_size + s2_val_size:]

print(f"Sentinel-1: {len(sent1_train)} train, {len(sent1_val)} val, {len(sent1_test)} test.")
print(f"Sentinel-2: {len(sent2_train)} train, {len(sent2_val)} val, {len(sent2_test)} test.")


# ---------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------

def translate_image_and_mask(tile_id, comp_dir, msk_dir):
    src = rasterio.open(os.path.join(comp_dir, tile_id))
    img = src.read().transpose(1, 2, 0).astype(np.float32)
    src.close()

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    src = rasterio.open(os.path.join(msk_dir, tile_id))
    mask = src.read(1)
    mask[mask > 1] = 0
    mask = to_categorical(mask, num_classes=2)
    src.close()

    return img, mask

def create_tf_dataset(df, comp_dir, msk_dir, batch_size, cache_path=None):
    def generator():
        for _, row in df.iterrows():
            yield translate_image_and_mask(row['tile_id'], comp_dir, msk_dir)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(128, 128, None), dtype=tf.float32),
            tf.TensorSpec(shape=(128, 128, 2), dtype=tf.float32)
        )
    )

    if cache_path:
        dataset = dataset.cache(cache_path)  # disque
    else:
        dataset = dataset.cache()  # RAM (carefull with large datasets)

    return dataset\
        .shuffle(200)\
        .batch(batch_size)\
        .prefetch(tf.data.AUTOTUNE)


# # convert the dataframe to a tf dataset using a generator (STOPS THE KILLED ERROR)
# def create_tf_dataset(df, comp_dir, msk_dir, batch_size):
#     def generator():
#         for i, row in df.iterrows():
#             yield translate_image_and_mask(row['tile_id'], comp_dir, msk_dir)
    
#     return tf.data.Dataset.from_generator(
#         generator, 
#         output_signature=(
#             tf.TensorSpec(shape=(128, 128, None), dtype=tf.float32),
#             tf.TensorSpec(shape=(128, 128, 2), dtype=tf.float32)
#         )
#     ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# easier way to do it but it will crash if too much tile are loaded

# def create_tf_dataset(df, comp_dir, msk_dir, batch_size):
#     images = []
#     masks = []

#     for i, row in df.iterrows():
#         img, mask = translate_image_and_mask(row['tile_id'], comp_dir, msk_dir)
#         images.append(img)
#         masks.append(mask)

#     X = np.array(images, dtype=np.float32)
#     y = np.array(masks, dtype=np.float32)

#     return tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)


train_dataset_sent1 = create_tf_dataset(
    sent1_train, sent1_dir, sent1_mask_dir, BATCH_SIZE,
    cache_path="./cache/cache_s1"
)

val_dataset_sent1 = create_tf_dataset(
    sent1_val, sent1_dir, sent1_mask_dir, BATCH_SIZE,
    cache_path="./cache/cache_s1_val"
)

train_dataset_sent2 = create_tf_dataset(
    sent2_train, sent2_dir, sent2_mask_dir, BATCH_SIZE,
    cache_path="./cache/cache_s2"
)

val_dataset_sent2 = create_tf_dataset(
    sent2_val, sent2_dir, sent2_mask_dir, BATCH_SIZE,
    cache_path="./cache/cache_s2_val"
)
test_dataset_sent1 = create_tf_dataset(sent1_test, sent1_dir, sent1_mask_dir, BATCH_SIZE)
test_dataset_sent2 = create_tf_dataset(sent2_test, sent2_dir, sent2_mask_dir, BATCH_SIZE)

# ---------------------------------------------------------
# Metrics functions
# ---------------------------------------------------------
#threshold = 0.5
# Correct pixel-wise metrics for Recall, Precision, F1
# def recall_m(y_true, y_pred):
#     y_true = y_true[..., 1]
#     y_pred = tf.cast(y_pred[..., 1] > threshold, tf.float32)
#     return tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_true) + 1e-7)

# def precision_m(y_true, y_pred):
#     y_true = y_true[..., 1]
#     y_pred = tf.cast(y_pred[..., 1] > threshold, tf.float32)
#     return tf.reduce_sum(y_true * y_pred) / (tf.reduce_sum(y_pred) + 1e-7)

# # bugged for sent-1
# def f1(y_true, y_pred):
#     y_true = y_true[..., 1]
#     y_pred = y_pred[..., 1]

#     tp = tf.reduce_sum(y_true * y_pred)
#     fp = tf.reduce_sum((1 - y_true) * y_pred)
#     fn = tf.reduce_sum(y_true * (1 - y_pred))

#     return 2*tp / (2*tp + fp + fn + 1e-7)
init_threshold = 0.5

def get_binary_masks(y_true, y_pred):
    y_true = y_true[..., 1]
    y_pred = tf.cast(y_pred[..., 1] > init_threshold, tf.float32)
    return y_true, y_pred



def iou(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.cast(y_pred[..., 1] > init_threshold, tf.int32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    return tp / (tp + fp + fn + 1e-7)




def dice_water(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.cast(y_pred[..., 1] > init_threshold, tf.float32)

    y_true = tf.cast(y_true == 1, tf.float32)
    y_pred = tf.cast(y_pred == 1, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    return 2 * tp / (2 * tp + fp + fn + 1e-7)

def dice_non_water(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.cast(y_pred[..., 1] > init_threshold, tf.float32)

    y_true = tf.cast(y_true == 0, tf.float32)
    y_pred = tf.cast(y_pred == 0, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    return 2 * tp / (2 * tp + fp + fn + 1e-7)

def weighted_f1(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.cast(y_pred[..., 1] > init_threshold, tf.int32)

    f1_scores = []
    supports = []

    for cls in [0, 1]:
        y_true_c = tf.cast(y_true == cls, tf.float32)
        y_pred_c = tf.cast(y_pred == cls, tf.float32)

        tp = tf.reduce_sum(y_true_c * y_pred_c)
        fp = tf.reduce_sum((1 - y_true_c) * y_pred_c)
        fn = tf.reduce_sum(y_true_c * (1 - y_pred_c))

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)

        f1_c = 2 * precision * recall / (precision + recall + 1e-7)
        support = tf.reduce_sum(y_true_c)

        f1_scores.append(f1_c)
        supports.append(support)

    f1_scores = tf.stack(f1_scores)
    supports = tf.stack(supports)

    return tf.reduce_sum(f1_scores * supports) / (tf.reduce_sum(supports) + 1e-7)

# ---------------------------------------------------------
# Model initialization
# ---------------------------------------------------------

custom_metrics = [
    "categorical_accuracy",
    weighted_f1, iou, dice_water, dice_non_water
]

#weights are unsused for categorical focal loss

# def compute_class_weights(df, msk_dir):
#     total = 0
#     counts = np.zeros(2)

#     for i, row in df.iterrows():
#         src = rasterio.open(os.path.join(msk_dir, row['tile_id']))
#         mask = src.read(1)
#         src.close()

#         for cls in [0, 1]:
#             counts[cls] += np.sum(mask == cls)

#         total += mask.size

#     weights = total / (2 * counts)
#     weights = weights / np.min(weights)  # normalize
#     weights[1] = weights[1]

#     return weights

# weights_s1 = compute_class_weights(sent1_train, sent1_mask_dir)
# print("Weights S1:", weights_s1) #debug print
# weights_s2 = compute_class_weights(sent2_train, sent2_mask_dir)
# print("Weights S2:", weights_s2) #debug print
weights_s1 = [1.0, 1.0]
weights_s2 = [1.0, 1.0]

# opt = tf.keras.optimizers.Adam(learning_rate=1e-4) # lower learning rate for better convergence with the focal loss, but it will take more time to train

model_sent1 = unet_model(tile_width=128, 
                         tile_height=128, 
                         n_bands=2, 
                         n_classes=2, 
                         n_blocks=5,
                         metrics=custom_metrics,
                         class_weight_list=weights_s1,
                         loss_function="categorical_focal_crossentropy",
                         #loss_function="cfce_focal_tversky",
                         #loss_function="focal_tversky",
                         # optimizer=opt #not necessary
                         )

#weight_path_sent1 = os.path.join(s1_model_dir, 'unet/1/model_weights.hdf5')
#model_sent1.load_weights(weight_path_sent1)

model_sent2 = unet_model(tile_width=128,
                        tile_height=128,
                        n_bands=9,
                        n_classes=2,
                        n_blocks=5,
                        metrics=custom_metrics,
                        class_weight_list=weights_s2,
                        loss_function="categorical_focal_crossentropy", 
                        #loss_function="cfce_focal_tversky",
                        #loss_function="focal_tversky",
                        )
#weight_path_sent2 = os.path.join(s2_model_dir, 'unet/1/model_weights.hdf5')
#model_sent2.load_weights(weight_path_sent2)

# ---------------------------------------------------------
# Training with TensorBoard
# --------------------------------------------------------- 

# Callbacks S1
# could stop before the best model for sent-1
early_stop_s1 = tf.keras.callbacks.EarlyStopping(
    monitor='val_weighted_f1',
    mode='max',
    patience=12,
    restore_best_weights=True,
    verbose=1
)
checkpoint_s1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{s1_save_dir}/best_model.keras",
    monitor='val_weighted_f1',
    mode='max',
    save_best_only=True,
    verbose=1

)
tensorboard_callback_sent1 = tf.keras.callbacks.TensorBoard(
    log_dir=sent1_log_dir,
    update_freq="epoch",
    write_graph=True,
    histogram_freq=0 # 1 if more RAM
)

# Callbacks S2
# could stop before the best model for sent-2
early_stop_s2 = tf.keras.callbacks.EarlyStopping(
    monitor='val_weighted_f1',
    mode='max',
    patience=12,
    restore_best_weights=True,
    verbose=1
)
checkpoint_s2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=f"{s2_save_dir}/best_model.keras",
    monitor='val_weighted_f1',
    mode='max',
    save_best_only=True,
    verbose=1
)
tensorboard_callback_sent2 = tf.keras.callbacks.TensorBoard(
    log_dir=sent2_log_dir,
    update_freq="epoch",
    write_graph=True,
    histogram_freq=0 # 1 if more RAM
)

print("\n--- Starting training on Sentinel-1 subset ---")

history_sent1 = model_sent1.fit(
    train_dataset_sent1,
    validation_data=val_dataset_sent1,
    epochs=1, #never put 0 put 1 instead if you want to skip
    callbacks=[tensorboard_callback_sent1, checkpoint_s1],
    verbose=2
)

#debug for flood ratio

# for batch in val_dataset_sent1.take(1):
#     x, y = batch
#     pred = model_sent1.predict(x)

#     print("Pred flood ratio:", np.mean(np.argmax(pred, axis=-1)))

print(f"\nTraining completed. TensorBoard logs are saved in: {logs_dir}/sent1") 

print("\n--- Starting training on Sentinel-2 subset ---")
history_sent2 = model_sent2.fit(
    train_dataset_sent2,
    validation_data=val_dataset_sent2,
    epochs=40, #never put 0 put 1 instead if you want to skip
    callbacks=[tensorboard_callback_sent2, checkpoint_s2],
    verbose=2,
)
print(f"\nTraining completed. TensorBoard logs are saved in: {logs_dir}/sent2")


# ---------------------------------------------------------
# Evaluation and visualization after training
# ---------------------------------------------------------



output_dir = f"./save/runs/{run_id}/predictions"
visualization_dir = f"./save/runs/{run_id}/visualizations"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(visualization_dir, exist_ok=True)

composite_dirs = {
    'Sentinel-1': sent1_dir,
    'Sentinel-2': sent2_dir
}
mask_dirs = {
    'Sentinel-1': sent1_mask_dir,
    'Sentinel-2': sent2_mask_dir
}

def evaluate_model(val_df, dataset, model, is_s2=False, max_viz=5, max_eval=100, threshold=[0.5,0.4,0.3]):
    print(f"\n--- Evaluating {dataset} on {max_eval} tiles ---")

    results = []
    for t in threshold: 
        viz_count = 0
        all_metrics = []

        for i, (_, row) in enumerate(val_df.head(max_eval).iterrows()):
            tile_id = row['tile_id']
            metrics = run_inference(tile_id, dataset, model, composite_dirs, mask_dirs, output_dir, score_threshold=t, with_gt=True)
            if metrics:
                all_metrics.append(metrics)

            if viz_count < max_viz:
                fig = visualize_tile(tile_id, dataset, composite_dirs, output_dir, mask_dirs, is_s2=is_s2, with_gt=True)
                fig.savefig(f"{visualization_dir}/{dataset}_{tile_id.split('.')[0]}.png", dpi=100, bbox_inches='tight')
                plt.close(fig)
                viz_count += 1

        df = pd.DataFrame(all_metrics)
        tp = df['TP'].sum()
        fp = df['FP'].sum()
        fn = df['FN'].sum()
        tn = df['TN'].sum()
        total = tp + fp + fn + tn

        precision   = tp / (tp + fp + 1e-7)
        recall      = tp / (tp + fn + 1e-7)
        f1_eau      = 2 * precision * recall / (precision + recall + 1e-7)
        iou         = tp / (tp + fp + fn + 1e-7)
        accuracy    = (tp + tn) / total

        support_eau     = tp + fn
        support_non_eau = tn + fp
        p0 = tn / (tn + fn + 1e-7)
        r0 = tn / (tn + fp + 1e-7)
        f1_non_eau  = 2 * p0 * r0 / (p0 + r0 + 1e-7)
        weighted_f1 = (f1_eau * support_eau + f1_non_eau * support_non_eau) / total

        mean_metrics = {
            'Accuracy': accuracy, 'Weighted F1': weighted_f1,
            'Dice Water': f1_eau, 'Dice Non-Water': f1_non_eau, 'Precision': precision,
            'Recall': recall, 'IoU': iou, '===threshold====': t, "water true positives": tp, "water false positives": fp, "water false negatives": fn, "water true negatives": tn
        }

        print(f"\n{dataset} — Global Metrics  pixel-wise:")
        for k, v in mean_metrics.items():
            print(f"  {k}: {v:.4f}")
        results.append(mean_metrics)
    best = max(results, key=lambda x: x['Weighted F1'])
    print("\nBest threshold:", best['===threshold===='])
    return results

model_sent1.load_weights(f"{s1_save_dir}/best_model.keras")
model_sent2.load_weights(f"{s2_save_dir}/best_model.keras")
thresholds = [0.5, 0.45, 0.4,0.35]
if(history_sent1 is not None):
    metrics_s1 = evaluate_model(sent1_test, 'Sentinel-1', model_sent1, is_s2=False, max_viz=10, threshold=thresholds)
if(history_sent2 is not None):
    metrics_s2 = evaluate_model(sent2_test, 'Sentinel-2', model_sent2, is_s2=True,  max_viz=10, threshold=thresholds, max_eval=len(sent2_test))

os.system('rm -rf cache')