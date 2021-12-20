# -*- coding: utf-8 -*-

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

class ConvAutoEncoder:
    @staticmethod
    def build(width, height, depth, filters=(128,), latent_dim=48):
        input_shape = (height, width, depth)
        channel_dim = -1
        inputs = layers.Input(shape=input_shape)
        x = inputs
        # Encoder layer
        for f in filters:
            x = layers.Conv2D(f, (3, 3), strides=2, padding='same')(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = layers.BatchNormalization(axis=channel_dim, name='enc_filter_' + str(f))(x)
        volume_size = K.int_shape(x)
        x = layers.Flatten()(x)
        # Latent layer
        latent = layers.Dense(latent_dim, name="encoded")(x)
        # Decoder layer
        x = layers.Dense(np.prod(volume_size[1:]))(latent)
        x = layers.Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
        # Reverse on decoder
        for f in filters[::-1]:
            x = layers.Conv2DTranspose(f, (3, 3), strides=2, padding='same')(x)
            x = layers.LeakyReLU(alpha=0.2)(x)
            x = layers.BatchNormalization(axis=channel_dim, name='dec_filter_' + str(f))(x)
        x = layers.Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = layers.Activation("sigmoid", name="decoded")(x)
        auto_encoder = Model(inputs, outputs, name="auto_encoder")
        return auto_encoder