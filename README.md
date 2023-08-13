# CQSIGN: Topological Contracted Quantized SIGN Framework
This repository is part of "Affordable Graph Neural Network Framework using Topological Graph Contraction" workshop paper published in MICCAI workshop ([MiLLAND 2023](https://miccaimilland.wixsite.com/milland2023)). This repository contains the main source code for our proposed CQSIGN framework along with other implementation of topological graph contraction on various memory-efficient GNN models (e.g., SIGN, QSIGN, GCN, Cluster-GCN, and GNN AutoScale). However, since some of the included model have different dependencies (e.g., different version of NetworkX), this repository will be organized as follows:

<img width="600" alt="Screenshot 2023-08-13 at 17 54 30" src="https://github.com/christopheradnel414/CQSIGN/assets/41734037/20a0a875-7cc0-4e65-909e-73b7f0ba5aba">

The main directory of CQSIGN consists of our proposed centrality-based topological graph contraction and the implementation of C-QSIGN, C-SIGN, and C-GCN models (including the [ActNN](https://github.com/ucbrise/actnn) quantization package). Additionally, the main directory also has 2 data generator scripts which generates contracted graphs of our datasets (e.g., PPI, Organ-C, and Organ-S) in a format that is readable to [Cluster-GCN](https://github.com/google-research/google-research/tree/master/cluster_gcn) and [GNN AutoScale (GAS)](https://github.com/rusty1s/pyg_autoscale). Hence, once the graph has been contracted, the included Cluster-GCN and GAS models can be executed with their respective original dependency, independent of our graph contraction module.

# Dependencies
Here, we have 3 different sets of package dependencies for the main CQSIGN directory (includes C-QSIGN, C-SIGN, C-GCN), Cluster-GCN directory, and GNN AutoScale directory, respectively. For the CQSIGN directory, we worked the following dependencies:
```
torch-scatter==2.1.1+pt20cu118
torch-sparse==0.6.17+pt20cu118
torch-geometric==2.3.0
torchvision==0.15.1+cu118
torch==2.0.0+cu118
ninja==1.11.1
numpy==1.24.1
networkx==3.0
scikit-learn==1.2.2
scipy==1.10.1
```
Next, for the included Cluster-GCN package, we worked with the following dependencies:
```
nvidia-tensorflow==1.15.5+nv22.12
metis==0.2a5
numpy==1.23.5
networkx==1.11
scikit-learn==1.2.2
scipy==1.10.1
setuptools
```
Finally, for the included GNN AutoScale package, we worked with the following dependencies:
```
torch-scatter==2.1.1+pt20cu118
torch-sparse==0.6.17+pt20cu118
torch-geometric==2.3.0
torch==2.0.0+cu118
hydra-core==1.3.2
numpy==1.24.1
scikit-learn==1.2.2
scipy==1.10.1
setuptools
```

# Setup

# Executing C-QSIGN, C-SIGN, and C-GCN

# Executing C-ClusterGCN

# Executing C-GAS (GNN AutoScale)

