# Res-UNet model: UNet with residual blocks in encoder and decoder
# Residual connections help gradient flow and allow deeper networks

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
    Add,
)

from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K

import tensorflow_addons as tfa
from tensorflow.keras import optimizers

import tensorflow as tf

from tensorflow.keras.regularizers import l2

import numpy as np

from losses import *


def res_conv_block(
    inputs, n_filters, dropout_prob=0, max_pooling=True, w_decay=0, norm=True
):
    """
    Residual convolutional downsampling block.

    Adds a residual (shortcut) connection from the block input to its output.
    A 1x1 Conv projection is used when the number of channels differs.

    Arguments:
        inputs        -- Input tensor
        n_filters     -- Number of filters for the convolutional layers
        dropout_prob  -- Dropout probability
        max_pooling   -- Use MaxPooling2D to reduce spatial dimensions
        w_decay       -- L2 regularization weight decay
        norm          -- Apply BatchNormalization before the first conv
    Returns:
        next_layer, skip_connection -- Next layer and skip connection outputs
    """
    x = inputs
    if norm:
        x = BatchNormalization()(x)

    x = Conv2D(
        n_filters,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(w_decay),
    )(x)
    x = Conv2D(
        n_filters,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(w_decay),
    )(x)

    if dropout_prob > 0:
        x = Dropout(dropout_prob)(x)

    # --- Shortcut / residual projection ---
    # Project input to n_filters channels with a 1x1 conv so shapes match
    shortcut = Conv2D(
        n_filters,
        1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=l2(w_decay),
    )(inputs)

    # Residual addition
    conv = Add()([x, shortcut])


    if max_pooling:
        next_layer = MaxPooling2D(2, strides=2)(conv)
    else:
        next_layer = conv

    skip_connection = conv

    return next_layer, skip_connection


def res_upsampling_block(expansive_input, contractive_input, n_filters, norm=False):
    """
    Residual convolutional upsampling block.

    After the transpose conv + skip concatenation, a residual connection
    is added around the two Conv2D layers.

    Arguments:
        expansive_input   -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters         -- Number of filters for the convolutional layers
        norm              -- Apply BatchNormalization after concatenation
    Returns:
        conv -- Tensor output
    """
    up = Conv2DTranspose(
        n_filters, 3, strides=2, padding="same"
    )(expansive_input)

    merge = concatenate([up, contractive_input], axis=3)

    if norm:
        merge = BatchNormalization()(merge)


    x = Conv2D(
        n_filters,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(merge)
    x = Conv2D(
        n_filters,
        3,
        activation="relu",
        padding="same",
        kernel_initializer="he_normal",
    )(x)

    #  Shortcut: project merged tensor to n_filters channels 
    shortcut = Conv2D(
        n_filters,
        1,
        padding="same",
        kernel_initializer="he_normal",
    )(merge)

    conv = Add()([x, shortcut])

    return conv


def resunet_model(
    n_classes,
    tile_width,
    tile_height,
    n_bands,
    n_blocks,
    class_weight_list=[1,1],
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
    Res-UNet: UNet with residual blocks in both encoder and decoder paths.

    Architecture:
        - Encoder: n_blocks residual conv blocks with max pooling (except last block)
        - Decoder: (n_blocks - 1) residual upsampling blocks with skip connections
        - Output: 1x1 Conv2D with softmax activation

    Arguments:
        n_classes          -- Number of output classes
        tile_width         -- Input tile width
        tile_height        -- Input tile height
        n_bands            -- Number of input bands/channels
        n_blocks           -- Total number of encoder blocks (including bottleneck)
        class_weight_list  -- Per-class weights for weighted loss functions
        n_filters_start    -- Number of filters in the first encoder block
        w_decay            -- L2 regularization weight decay
        droprate           -- Base dropout rate
        drop_multiplier    -- Per-block dropout multipliers (list of length n_blocks)
        weight_multiplier  -- Per-block weight decay multipliers (list of length n_blocks)
        filter_growth      -- Filter growth factor between blocks (default: 2 = doubles)
        normalize_inputs   -- Apply BatchNormalization in encoder blocks
        optimizer          -- Keras optimizer (string or optimizer instance)
        loss_function      -- Loss function key (see losses_dict) or Keras loss string
        metrics            -- List of metrics to track during training
    Returns:
        model -- Compiled Keras Model
    """
    inputs = Input((tile_width, tile_height, n_bands))
    x = inputs
    n_filters = n_filters_start
    contracting_blocks = []

    if drop_multiplier is None:
        drop_multiplier = [1.0] * n_blocks

    if weight_multiplier is None:
        weight_multiplier = [1.0] * n_blocks

    # Encoder 
    for i in range(n_blocks):
        if normalize_inputs:
            norm = False if i == 0 else True
        else:
            norm = False

        if i < n_blocks - 1:
            # Downsampling block
            x, skip = res_conv_block(
                inputs=x,
                n_filters=n_filters,
                w_decay=w_decay * weight_multiplier[i],
                dropout_prob=droprate * drop_multiplier[i],
                norm=norm,
            )
            contracting_blocks.append(skip)
        else:
            # Bottleneck: no max pooling, no batch norm, skip connection discarded
            x, _ = res_conv_block(
                inputs=x,
                n_filters=n_filters,
                w_decay=w_decay * weight_multiplier[i],
                dropout_prob=droprate * drop_multiplier[i],
                max_pooling=False,
                norm=False,
            )

        n_filters *= filter_growth

    # Decoder
    contracting_blocks.reverse()
    n_filters //= filter_growth

    for i in range(n_blocks - 1):
        norm = False
        if normalize_inputs:
            norm = True
            if i == n_blocks - 2:
                norm = False

        n_filters //= filter_growth
        x = res_upsampling_block(
            x, contracting_blocks[i], n_filters=n_filters, norm=norm
        )

    # Output layer
    outputs = Conv2D(n_classes, 1, activation="softmax", padding="same")(x)

    model = Model(inputs=inputs, outputs=outputs)

    # Loss function selection
    class_weights = class_weight_list

    weighted_dice_loss = WeightedDiceLoss(class_weights)
    weighted_jaccard_loss = WeightedJaccardLoss(class_weights)
    focal_tversky_loss = FocalTverskyLoss()
    alpha = class_weight_list
    weighted_cfce = CategoricalFocalCrossentropy(alpha=alpha)

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
        metrics=metrics,
    )

    return model
