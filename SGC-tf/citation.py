#####################################################################################################
# File Name: models.py
# Author: Upinder Kaur
# Purpose: Main file for training and testing SGC using tensorflow based code. The GCN implementation
# is similar to that given in Spektral example code, but modified as was published in the PyTorch
# implementation of GCN and SGC.
#####################################################################################################

import time
import argparse
import numpy as np
from utils import *
from models import *
from time import perf_counter
from tensorflow.python.client import device_lib
from spektral.datasets import citation
#check and select device
device_name = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
if len(device_name) != 0:
    device_name = "/gpu:0"
    print('GPU')
else:
    print('CPU')
    device_name = "/cpu:0"

# Arguments
args = get_args()
epoch, lr, weight_decay = args.epochs, args.lr, args.weight_decay
K=2
#load dataset
A, X, y, train_mask, val_mask, test_mask = citation.load_data(args.dataset)
X=X.toarray()
n = X.shape[0] #number of nodes
nf = X.shape[1] #numder of node features
nc = y.shape[1] #number of classes
#pre-processing steps
fltr = GraphConv.preprocess(A).astype('f4') #for gcn implementation
print(fltr.shape)
A = adj_normalization(A)
X = row_normalize(X)
print(X.shape)
#pre-computation of K-step graph propagation
time_init = perf_counter()
X_p = precomputation(X,A,K)
time_compute = perf_counter() - time_init
print("time for pre-computation:",time_compute)

#cast to tensorflow tensor
X = tf.cast(X, dtype=tf.float32)

X_p = tf.cast(X_p, dtype=tf.float32)
#initialize model
if args.model == "SGC":
    print("Starting SGC model", args.model)
    with tf.device(device_name):
        sgc_model = model_SGC(nc, nf, lr,weight_decay)
if args.model == "GCN":
    gcn_model = model_GCN(args.hidden,n,nc, nf, args.lr, args.weight_decay)
#train model
if args.model == "SGC":
    validation_data = (X_p[val_mask], y[val_mask])
    with tf.device(device_name):
        time_init = perf_counter()
        sgc_model.fit(X_p[train_mask],
          y[train_mask],
          epochs= epoch,
          batch_size=n,
          validation_data=validation_data,
          shuffle=False, 
          callbacks=[
              EarlyStopping(patience=200,  restore_best_weights=True)
          ])
        time_compute = perf_counter() - time_init
        print("time for training:",time_compute)
if args.model == "GCN":
    validation_data = ([X,fltr], y, val_mask)
    with tf.device(device_name):
      gcn_model.fit([X, fltr],
          y,
          epochs=epoch,
          sample_weight=train_mask,
          batch_size=n,
          validation_data=validation_data,
          shuffle=False, 
          callbacks=[
              EarlyStopping(patience=200,  restore_best_weights=True)
          ])
#evaluate model on test set
if args.model == "SGC":
    with tf.device(device_name):
        sgc_evaluation = sgc_model.evaluate(X_p[test_mask],
                              y[test_mask],
                              batch_size=n)
        print('Done.\n'
         'Test loss: {}\n'
          'Test accuracy: {}'.format(*sgc_evaluation))
if args.model == "GCN":
    with tf.device(device_name):
        gcn_evaluation = gcn_model.evaluate([X,fltr],
                              y,sample_weight=test_mask,
                              batch_size=n)
        print('Done.\n'
         'Test loss: {}\n'
          'Test accuracy: {}'.format(*gcn_evaluation))
