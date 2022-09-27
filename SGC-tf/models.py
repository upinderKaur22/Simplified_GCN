#####################################################################################################
# File Name: models.py
# Author: Upinder Kaur
# Purpose: TensorFlow models of SGC and GCN
#####################################################################################################
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from spektral.layers import GraphConv

def model_SGC(nc,nf,learning_rate, l2_reg):
    """
    A Simple TensorFlow Implementation of SGC.
    Assuming the features have been preprocessed with k-step graph propagation.
    """
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(nf,)))
    model.add(tf.keras.layers.Dense(nc,activation='softmax',
                   kernel_regularizer=l2(l2_reg),
                   use_bias=False))
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
    model.summary()
    return model

def model_GCN(channels,n, n_classes,F,l2_reg, learning_rate):#derived from the original implementation of GCN by Kipf and Spektral based examples
    """
    A Two-layer GCN.
    """
    print(n, F)
    Xi = Input(shape=(F, ))
    flt = Input((n, ), sparse=True)
    gc1 = GraphConv(16, activation='relu', kernel_regularizer = l2(l2_reg), use_bias=False)([Xi,flt])
    drop = Dropout(0.5)(gc1)
    gc2 = GraphConv(n_classes, activation='relu', kernel_regularizer = l2(l2_reg), use_bias=False)([drop, flt])
    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[Xi,flt], outputs=gc2)
    model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
    model.summary()
    return model
