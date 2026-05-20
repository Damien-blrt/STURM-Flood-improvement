from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose,
    MaxPooling2D, concatenate,
    BatchNormalization, Activation,
    Add
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from losses import *


def residual_block(x, filters, weight_decay=1e-4, stride=1):
    """
    Standard residual block used in both encoder and decoder.

    Structure:
        BN -> ReLU -> Conv3x3
        BN -> ReLU -> Conv3x3
        + shortcut connection (identity or projection)

    If the number of filters changes, a 1x1 convolution
    is applied to the shortcut to match dimensions.
    """

    shortcut = x

    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, 3, strides=stride, padding="same",
               kernel_initializer="he_normal",
               kernel_regularizer=l2(weight_decay))(x)


    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, 3, padding="same",
               kernel_initializer="he_normal",
               kernel_regularizer=l2(weight_decay))(x)


    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding="same",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(weight_decay))(shortcut)

    x = Add()([x, shortcut])

    return x


def encoder_block(x, filters, weight_decay=1e-4):
    """
    Encoder block:
        - Residual feature extraction
        - Skip connection stored for decoder
        - Spatial downsampling via MaxPooling
    """
    x = residual_block(x, filters, weight_decay)
    skip = x
    x = MaxPooling2D(2)(x)
    return x, skip


def decoder_block(x, skip, filters, weight_decay=1e-4):
    """
    Decoder block:
        - Upsampling via transposed convolution
        - Concatenation with encoder skip connection
        - Residual refinement block
    """

    x = Conv2DTranspose(filters, 2, strides=2, padding="same")(x)

    x = concatenate([x, skip])

    x = residual_block(x, filters, weight_decay)

    return x


def resunet_model(
    n_classes,
    tile_width,
    tile_height,
    n_bands,
    n_blocks,
    class_weight_list=[1, 1],
    n_filters_start=64,
    w_decay=1e-5,
    droprate=0.3,
    drop_multiplier=None,
    weight_multiplier=None,
    filter_growth=2,
    normalize_inputs=True,
    optimizer="adam",
    loss_function="categorical_crossentropy",
    metrics=["categorical_accuracy"],
):
    """
    ResUNet model (UNet + residual blocks)

    Architecture:
        Encoder: residual blocks + pooling
        Bottleneck: residual block
        Decoder: upsampling + skip connections + residual blocks
        Output: pixel-wise softmax segmentation
    """

    inputs = Input((tile_width, tile_height, n_bands))
    x = inputs

    skips = []

    if drop_multiplier is None:
        drop_multiplier = [1.0] * n_blocks
    if weight_multiplier is None:
        weight_multiplier = [1.0] * n_blocks

    n_filters = n_filters_start

    for i in range(n_blocks):
        x, skip = encoder_block(x, n_filters, w_decay * weight_multiplier[i])
        skips.append(skip)
        n_filters *= filter_growth

    
    x = residual_block(x, n_filters, w_decay)

   
    skips = skips[::-1]

    for i in range(n_blocks):
        n_filters //= filter_growth
        x = decoder_block(x, skips[i], n_filters, w_decay)

   
    outputs = Conv2D(n_classes, 1, activation="softmax")(x)

    model = Model(inputs, outputs)


    weighted_dice_loss = WeightedDiceLoss(class_weight_list)
    weighted_jaccard_loss = WeightedJaccardLoss(class_weight_list)
    focal_tversky_loss = FocalTverskyLoss()
    weighted_cfce = CategoricalFocalCrossentropy(alpha=class_weight_list)

    losses_dict = {
        "categorical_focal_crossentropy": cfce,
        "weighted_categorical_focal_crossentropy": weighted_cfce,
        "tversky_loss": tversky_loss,
        "dice_loss": dice_loss,
        "weighted_dice_loss": weighted_dice_loss,
        "cce_dice": cce_dice_loss,
        "focal_dice": cfce_dice_loss,
        "jaccard_loss": jaccard_loss,
        "weighted_jaccard_loss": weighted_jaccard_loss,
        "cce_jaccard": cce_jaccard_loss,
        "focal_jaccard": cfce_jaccard_loss,
        "focal_tversky": focal_tversky_loss,
        "cfce_focal_tversky": cfce_focal_tversky_loss,
    }

    loss = losses_dict.get(loss_function, f"{loss_function}")

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model