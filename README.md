# CQSIGN: Topological Contracted Quantized SIGN Framework
This repository is part of "Affordable Graph Neural Network Framework using Topological Graph Contraction" workshop paper published in MICCAI workshop ([MiLLAND 2023](https://miccaimilland.wixsite.com/milland2023)). This repository contains the main source code for our proposed CQSIGN framework along with other implementation of topological graph contraction on various memory-efficient GNN models (e.g., SIGN, QSIGN, GCN, Cluster-GCN, and GNN AutoScale). However, since some of the included model have different dependencies (e.g., different version of NetworkX), this repository will be organized as follows:

<img width="600" alt="Screenshot 2023-08-13 at 17 44 29" src="https://github.com/christopheradnel414/CQSIGN/assets/41734037/0ba01fdd-fd33-445a-99a4-3213fb6c5283">

The main directory of CQSIGN consists of our proposed centrality-based topological graph contraction and the implementation of C-QSIGN, C-SIGN, and C-GCN models. Additionally, the main directory also has 2 data generator scripts which generates contracted graphs of our datasets (e.g., PPI, Organ-C, and Organ-S) in a format that is readable to [Cluster-GCN](https://github.com/google-research/google-research/tree/master/cluster_gcn) and [GNN AutoScale (GAS)](https://github.com/rusty1s/pyg_autoscale). Hence, once the graph has been contracted, the included Cluster-GCN and GAS models can be executed with their respective original dependency, independent of our graph contraction module.


