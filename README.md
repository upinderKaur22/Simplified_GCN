## Simplifying Graph Convolutional Networks: Comprehension and Implementation

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

#### Author: 
* [Upinder Kaur]


### Overview
This repository contains an implementation of the Simple Graph Convolution (SGC) model, as described in the ICML2019 paper [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153) using TensorFlow-based code. The original repository of the code (implemented using PyTorch) is available at:
(https://github.com/Tiiiger/SGC)

This repository has two main folder: SGC-tf and ML-SGC. It also contains a notebook file (combined-execution-term_paper.ipynb). 

### SGC-tf :Citation Networks Experiments
The SGC-tf contains Tensorflow based code of the SGC model in 3 files: model.py, utils.py, and citation.py. All three files are written from scratch by ourselves.
Only two utility function: row normalize and adj_normalization are similar in code to most graph network implementations. This code executes
the SGC model for citation datasets Cora, Citeseer, and Pubmed. The metrics of comparison are test accuracy and training time.

The arguments that can be changed by the user are: 
- epochs; default =100; number of epochs for training
- lr; default=0.2 ; learning rate
- weight_decay; default=5e-6
- dataset; default=cora
- model; default = SGC
These can be changed from the command line itself. When the code is executed, the datasets are downloaded using Spektral sources, for which you need Spektral
library installed. 

The original implementation of GCN by Kipf (in Spektral) is used for the 2-layer GCN model in these files
(link:https://github.com/danielegrattarola/spektral/tree/master/examples). 

### ML-SGC: Multi-label Image Recognition Experiments
The extension of SGC to the task of Multi-label Image recognition based on the framework proposed by Chen et at. 2019 (https://arxiv.org/abs/1904.03582)
was completed by modifying their original code repository. The repository was modified at 3 main locations: engine.py, demo_voc2007_gcn.py,models.py and util.py.
The original repository can be sourced at: (https://github.com/Megvii-Nanjing/ML-GCN). In this repository, we added the SGC model in models.py, created the
SGC engine for the learning and evalution in engine.py, and modified the original algorithm for training and evaluation in demo_voc_2007_gcn.py. The precomputation
function was added to utils.py. We also added new command line argument choices,so that the user can switch between the GCN and SGC implementation using the
same code. Some of the command line choices are:
- model_choice; default = GCN; choose between [GCN, SGC]
- image-size; default = 224
- data; path to dataset
- batch-size; default = 16
- j (number of workers); default = 8
- epochs; default = 100
- momentum; default = 0.9
- weight_decay; default = 1e-4
- resume; this needs a path to the location of the stored checkpoint.pth.tar file
- e; default False; for evaluation


### Dependencies
If working with Google Colab, you only need to install 2 extra libraries: Spektral and Torchnet. 
Otherwise, to execute this code you need:
- Tensorflow
- Spektral
- Numpy
- Scipy
- tqdm
- time, sys
- torchnet
- torch-0.3.1
- torchvision-0.2.0


### Data
This code needs four datasets: Cora, Citeseer, PubMed, and PASCAL VOC 2007. The citation networks Cora, Pubmed and Citeseer are downloaded using
Spektral sources which ultimately download the code from the link: (https://github.com/tkipf/gcn/tree/master/gcn/data). 
The VOC2007 dataset was extracted from the tar file given by the authors under /data of the repository
(link: https://github.com/Megvii-Nanjing/ML-GCN/tree/master/data/voc). It is also a publicy availble dataset and can be sourced
from other locations as well.

As soon as the executions starts, the datasets will be downloaded/extracted into the memory.

### Usage
To allow for easier execution, we created a combined-execution-term_paper notebook. This is provided with the two subpacks of code. 
The initial cells of this notebook help setup path and download required libraries. 
Further, it will guide you on how to run each command with the options available. This notebook is especially useful if you are using
Google Colab.
A sample cell from the notebook:
```
#Now we are ready to execute. Datasets available are Cora, Pubmed and Citeseer. Arguments that you can set are:
# epochs; default =100; number of epochs for training
#lr; default=0.2 ; learning rate
#weight_decay; default=5e-6
#dataset; default=cora
#model; default = SGC
!python3 "citation.py" --dataset cora --weight_decay 1.3026973714043257e-05
```
If you do not want to use this notebook, 

For citation networks use:
```
$ python citation.py --dataset --cora
```
For ML-SGC runs use:
```
$ python demo_voc2007_gcn.py 'data/voc' --image-size 448 -j 2 --batch-size 8 --epochs 40 -model_choice SGC 
```



