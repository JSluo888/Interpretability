import os
import random
import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pysam
import anndata
import h5py
import matplotlib.pyplot as plt
import os
import psutil
import math
import pickle
import seaborn as sns
import scipy
import configargparse
import sys
import gc
import scanpy as sc
from datetime import datetime
import time
import pandas as pd
from Bio import SeqIO
from scipy import sparse
from evoaug_tf import augment, evoaug
from wandb.keras import WandbCallback
import wandb

##############
# activation #
##############
class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x. GELU approximation
        return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x

##########
# layers #
##########
class StochasticReverseComplement(tf.keras.layers.Layer):
    """Stochastically reverse complement a one hot encoded DNA sequence."""

    def __init__(self, **kwargs):
        super(StochasticReverseComplement, self).__init__()

    def call(self, seq_1hot, training=None):
        if training:
            rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
            rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
            reverse_bool = tf.random.uniform(shape=[]) > 0.5
            src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, tf.constant(False)


class SwitchReverse(tf.keras.layers.Layer):
    """Reverse predictions if the inputs were reverse complemented."""

    def __init__(self, **kwargs):
        super(SwitchReverse, self).__init__()

    def call(self, x_reverse):
        x = x_reverse[0]
        reverse = x_reverse[1]

        xd = len(x.shape)
        if xd == 3:
            rev_axes = [1]
        elif xd == 4:
            rev_axes = [1, 2]
        else:
            raise ValueError("Cannot recognize SwitchReverse input dimensions %d." % xd)

        return tf.keras.backend.switch(reverse, tf.reverse(x, axis=rev_axes), x)


class StochasticShift(tf.keras.layers.Layer):
    """Stochastically shift a one hot encoded DNA sequence."""

    def __init__(self, shift_max=0, pad="uniform", **kwargs):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.augment_shifts = tf.range(-self.shift_max, self.shift_max + 1)
        self.pad = pad

    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tf.random.uniform(
                shape=[], minval=0, dtype=tf.int64, maxval=len(self.augment_shifts)
            )
            shift = tf.gather(self.augment_shifts, shift_i)
            sseq_1hot = tf.cond(
                tf.not_equal(shift, 0),
                lambda: shift_sequence(seq_1hot, shift),
                lambda: seq_1hot,
            )
            return sseq_1hot
        else:
            return seq_1hot

    def get_config(self):
        config = super().get_config().copy()
        config.update({"shift_max": self.shift_max, "pad": self.pad})
        return config
        
def shift_sequence(seq, shift, pad_value=0.25):
    """Shift a sequence left or right by shift_amount.

    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if seq.shape.ndims != 3:
        raise ValueError("input sequence should be rank 3")
    input_shape = seq.shape

    pad = pad_value * tf.ones_like(seq[:, 0 : tf.abs(shift), :])

    def _shift_right(_seq):
        # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return tf.concat([pad, sliced_seq], axis=1)

    def _shift_left(_seq):
        # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return tf.concat([sliced_seq, pad], axis=1)

    sseq = tf.cond(
        tf.greater(shift, 0), lambda: _shift_right(seq), lambda: _shift_left(seq)
    )
    sseq.set_shape(input_shape)

    return sseq

filepath = '/home/tianhao/pbmc.h5'
with h5py.File(filepath, 'r') as dataset:
    x_train = np.array(dataset['X_train']).astype(np.float32)
    y_train = np.array(dataset['Y_train']).astype(np.float32)
    x_valid = np.array(dataset['X_valid']).astype(np.float32)
    y_valid = np.array(dataset['Y_valid']).astype(np.int32)
    x_test = np.array(dataset['X_test']).astype(np.float32)
    y_test = np.array(dataset['Y_test']).astype(np.int32)

def custom(input_shape):
    inputs = tf.keras.Input(shape=input_shape, name='sequence')

    (nn, reverse_bool,) = StochasticReverseComplement()(inputs)
    nn=StochasticShift(3)(nn)

    #Conv Block
    nn=GELU()(nn)
    nn=tf.keras.layers.Conv1D(filters=288,
                           kernel_size=17,
                           strides=1,
                           padding='same',
                           use_bias=False,
                           dilation_rate=1,
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l2(0),)(nn)
    nn=tf.keras.layers.BatchNormalization(momentum=0.90,gamma_initializer='ones')(nn)
    nn = tf.keras.layers.Activation('exponential', name='conv_activation')(nn)
    nn=tf.keras.layers.MaxPool1D(pool_size=3, padding='same')(nn)

    #Conv Tower
    def _round(x):
        return int(np.round(x / 1) * 1)
    rep_filters = 288
    for i in range(6):
        if i != 0:
            nn=GELU()(nn)
        nn=tf.keras.layers.Conv1D(filters=_round(rep_filters),
                           kernel_size=5,
                           strides=1,
                           padding='same',
                           use_bias=False,
                           dilation_rate=1,
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l2(0),)(nn)
        nn=tf.keras.layers.BatchNormalization(momentum=0.90,gamma_initializer='ones')(nn)
        nn=tf.keras.layers.MaxPool1D(pool_size=2, padding='same')(nn)
        rep_filters*=1.122

    #Conv Block
    nn=GELU()(nn)
    nn=tf.keras.layers.Conv1D(filters=256,
                           kernel_size=1,
                           strides=1,
                           padding='same',
                           use_bias=False,
                           dilation_rate=1,
                           kernel_initializer="he_normal",
                           kernel_regularizer=tf.keras.regularizers.l2(0),)(nn)
    nn=tf.keras.layers.BatchNormalization(momentum=0.90,gamma_initializer='ones')(nn)
    
    #Dense
    nn=GELU()(nn)
    _, seq_len, seq_depth = nn.shape
    nn = tf.keras.layers.Reshape((1, seq_len * seq_depth,))(nn)
    nn = tf.keras.layers.Dense(units=32, use_bias=False, kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l1_l2(0,0),)(nn)
    nn=tf.keras.layers.BatchNormalization(momentum=0.90,gamma_initializer='ones')(nn)
    nn = tf.keras.layers.Dropout(rate=0.2)(nn)

    nn=GELU()(nn)

    #Final
    nn=tf.keras.layers.Dense(units=y_train.shape[-1], use_bias=True, activation='sigmoid', 
                          kernel_initializer="he_normal",kernel_regularizer=tf.keras.regularizers.l1_l2(0,0))(nn)

    #Swtich back
    nn=SwitchReverse()([nn, reverse_bool])

    nn=tf.keras.layers.Flatten()(nn)
    model = tf.keras.Model(inputs=inputs, outputs=nn)
    model.summary()
    
    return model

N,L,A = x_train.shape

tf.keras.backend.clear_session()
model = custom(input_shape=(L,A))

def reinitialize_kernel_weights(model, kernel_initializer):
  # can change initialization after creating model this way!
  for i, layer in enumerate(model.layers):
    if hasattr(layer, 'kernel_initializer'):

      # Get the current configuration of the layer
      weight_initializer = layer.kernel_initializer

      val = layer.get_weights()
      if len(val) > 1:
        old_weights, old_biases = val
        layer.set_weights([kernel_initializer(shape=old_weights.shape), old_biases])
      else:
        old_weights = val[0]
        layer.set_weights([kernel_initializer(shape=old_weights.shape)])

kernel_initializer = tf.keras.initializers.RandomNormal(stddev=0.001)  # Replace this with your desired initializer
reinitialize_kernel_weights(model, kernel_initializer)

# compile model
loss_fn = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.95,beta_2=0.9995)
model.compile(loss=loss_fn, optimizer=optimizer,
                  metrics=[tf.keras.metrics.AUC(name='auc', curve='ROC', multi_label=True),
                           tf.keras.metrics.AUC(name='auc_pr', curve='PR', multi_label=True)])

import wandb as wb
from wandb.keras import WandbCallback

wb.init(project='FINAL', group = 'pbmc_exp' , name='10')

out_dir = '/home/tianhao/best_models'
filepath_best = '%s/pbmc_exp_valauc_final_10.h5'%out_dir
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath_best, 
                                       save_best_only=True,
                                       save_weights_only=True, 
                                       monitor='val_auc', mode='max',
                                      save_freq='epoch'),
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=1e-6,
                                     mode='max', patience=10, verbose=1,restore_best_weights=True),
    WandbCallback(monitor='val_auc', mode='max'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_auc',
                                         factor=0.2, patience=3,
                                         verbose=1, mode='max'),
]

    # train the model
history = model.fit(
        x_train,
        y_train,
        epochs=200,
        callbacks=callbacks,
        validation_data=(x_valid, y_valid))

