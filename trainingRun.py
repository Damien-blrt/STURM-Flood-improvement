import os
import datetime
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import sys


sys.path.append('./arch')
from model import unet_model

# ---------------------------------------------------------
# 1. Folders creation
# ---------------------------------------------------------


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
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(sent1_log_dir, exist_ok=True)
os.makedirs(sent2_log_dir, exist_ok=True)


# ---------------------------------------------------------
# Configuration


sent1_metadata = pd.read_csv(sent1_metadata_path)
sent2_metadata = pd.read_csv(sent2_metadata_path)
SEED = 1
SUBSET_SIZE = 20
BATCH_SIZE = 4 # to be able to train on a small subset
TRAINING_SIZE = int(SUBSET_SIZE * 0.8)
VALIDATION_SIZE = int(SUBSET_SIZE * 0.2)

sent1_subset = sent1_metadata.sample(n=SUBSET_SIZE, random_state=SEED)
sent2_subset = sent2_metadata.sample(n=SUBSET_SIZE, random_state=SEED)

sent1_train = sent1_subset.iloc[:TRAINING_SIZE]
sent1_val = sent1_subset.iloc[TRAINING_SIZE:]
sent2_train = sent2_subset.iloc[:TRAINING_SIZE]
sent2_val = sent2_subset.iloc[TRAINING_SIZE:]

#debug print ----------------------------------------------

print(f"Sentinel-1: {len(sent1_train)} for training, {len(sent1_val)} for validation.")
print(f"Sentinel-2: {len(sent2_train)} for training, {len(sent2_val)} for validation.")

# ---------------------------------------------------------

#translate the image and the mask to the format expected by the model (image: (128, 128, 2), mask: (128, 128, 2))
def translate_image_and_mask(tile_id, comp_dir, msk_dir):
    src = rasterio.open(os.path.join(comp_dir, tile_id))
    img = src.read().transpose(1, 2, 0)
    src.close()
    src = rasterio.open(os.path.join(msk_dir, tile_id))
    mask = src.read(1)
    mask[mask > 1] = 0
    mask = to_categorical(mask, num_classes=2) # one-hot encoding for the softmax output
    src.close()

    return img, mask


# convert the dataframe to a tf dataset
def create_tf_dataset(df, comp_dir, msk_dir, batch_size):
    images = []
    masks = []

    for i, row in df.iterrows():
        img, mask = translate_image_and_mask(row['tile_id'], comp_dir, msk_dir)
        images.append(img)
        masks.append(mask)

    X = np.array(images)
    y = np.array(masks)

    return tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)

train_dataset_sent1 = create_tf_dataset(sent1_train, sent1_dir, sent1_mask_dir, BATCH_SIZE)
val_dataset_sent1 = create_tf_dataset(sent1_val, sent1_dir, sent1_mask_dir, BATCH_SIZE)
train_dataset_sent2 = create_tf_dataset(sent2_train, sent2_dir, sent2_mask_dir, BATCH_SIZE)
val_dataset_sent2 = create_tf_dataset(sent2_val, sent2_dir, sent2_mask_dir, BATCH_SIZE)

# ---------------------------------------------------------
# Model initialization

model_sent1 = unet_model(tile_width=128, 
                         tile_height=128, 
                         n_bands=2, 
                         n_classes=2, 
                         n_blocks=5,
                         class_weight_list=[1.0, 1.0], # Equilibrated weights for the test
                         )

model_sent2 = unet_model(tile_width=128,
                        tile_height=128,
                        n_bands=9,
                        n_classes=2,
                        n_blocks=5,
                        class_weight_list=[1.0, 1.0], # Equilibrated weights for the test
                        )

# ---------------------------------------------------------
# Training with TensorBoard

tensorboard_callback_sent1 = tf.keras.callbacks.TensorBoard(
    log_dir=sent1_log_dir,
    update_freq="epoch",
    write_graph=True,
    histogram_freq=1
)

tensorboard_callback_sent2 = tf.keras.callbacks.TensorBoard(
    log_dir=sent2_log_dir,
    update_freq="epoch",
    write_graph=True,
    histogram_freq=1
)

print("\n--- Starting training on Sentinel-1 subset ---")

history_sent1 = model_sent1.fit(
    train_dataset_sent1,
    validation_data=val_dataset_sent1,
    epochs=10, 
    callbacks=[tensorboard_callback_sent1],
    verbose=1
)
print(f"\nTraining completed. TensorBoard logs are saved in: {logs_dir}/sent1") 

print("\n--- Starting training on Sentinel-2 subset ---")
history_sent2 = model_sent2.fit(
    train_dataset_sent2,
    validation_data=val_dataset_sent2,
    epochs=10,
    callbacks=[tensorboard_callback_sent2],
    verbose=1
)
print(f"\nTraining completed. TensorBoard logs are saved in: {logs_dir}/sent2") 