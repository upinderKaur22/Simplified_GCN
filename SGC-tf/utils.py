#####################################################################################################
# File Name: Utils.py
# Author: Upinder Kaur
# Purpose: supporting and utility functions for main implementation
#####################################################################################################
import numpy as np
import scipy.sparse as sp
import sys
import argparse
from time import perf_counter
from tensorflow.sparse import sparse_dense_matmul, SparseTensor

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=0,
                        help='Number of hidden units.')
    parser.add_argument('--dataset', type=str, default="cora",
                        help='Dataset to use.')
    parser.add_argument('--model', type=str, default="SGC",
                        choices=["SGC", "GCN"],
                        help='model to use.')
    args, _ = parser.parse_known_args()
    return args

def convert_sparse_matrix_to_sparse_tensor(sm):
    coo = sm.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return SparseTensor(indices, coo.data, coo.shape)

def precomputation(features, adj, degree):
  for i in range(degree):
    features = sp.csr_matrix.dot(adj, features)
  return features

def row_normalize(mx): #this function is similar to the row normalization function use by the original GCN implementation by Kipf (Github:tkipf/gcn)
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def adj_normalization(adj):#this function is similar to the adj normalization function use by the original GCN implementation by Kipf (Github:tkipf/gcn)
  adj = adj + sp.eye(adj.shape[0])
  adj = sp.coo_matrix(adj)
  rowsum = np.array(adj.sum(1))
  d_inv_sqrt = np.power(rowsum, -0.5).flatten()
  d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
  d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
  return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


