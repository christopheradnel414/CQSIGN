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
metis==0.2a5
hydra-core==1.3.2
numpy==1.24.1
scikit-learn==1.2.2
scipy==1.10.1
setuptools
```
Note that we are using CUDA 11.8 toolkit paired with an NVIDIA RTX 3050ti GPU with driver version 525.125.06. User are recommended to use the CUDA toolkit version that corresponds to their NVIDIA GPU driver. Details can be found [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver).

# Setup
Note that these instructions are written for Linux 22.04 with NVIDIA driver version 525.125.06.
## CQSIGN Directory Setup
1. To setup the main CQSIGN directory, user is recommended to create a new Python 3.9.16 virtual environment using [conda](https://conda.io/projects/conda/en/latest/index.html).
3. Install NVIDIA CUDA 11.8 toolkit from [here](https://developer.nvidia.com/cuda-11-8-0-download-archive). Depending on the user's NVIDIA driver version, different version of CUDA toolkit might be necessary.
4. Add the newly installed CUDA toolkit directory to bashrc by adding these lines to ~/.bashrc file:
```
# CUDA Toolkit 11.8
if [ -z $LD_LIBRARY_PATH ]; then
  LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64
else
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64
fi
export LD_LIBRARY_PATH
export PATH="/usr/local/cuda-11.8/bin:$PATH"
```
4. Install dependencies using the following pip commands:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-2.0.0+11.8.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-2.0.0+11.8.html
pip install torch-geometric
pip install ninja==1.11.1
pip install numpy==1.24.1
pip install networkx==3.0
pip install scikit-learn==1.2.2
pip install scipy==1.10.1
```
5. Install ActNN quantization package by going to ActNN folder and running the following script in cli:
```
pip install -v -e .
```
## Cluster-GCN Directory Setup
1. User is recommended to create a new Python 3.9.16 virtual environment for the Cluster-GCN dependencies.
2. Go to CQSIGN/OtherBenchmarks/Cluster-GCN/GKlib and install [GKlib](https://github.com/KarypisLab/GKlib) by executing the following script:
```
make
make install
```
3. CQSIGN/OtherBenchmarks/Cluster-GCN/METIS and install [METIS](https://github.com/KarypisLab/METIS) by executing the following script:
```
sudo apt-get install build-essential
sudo apt-get install cmake

make config shared=1 cc=gcc prefix=~/local
make install
```
4. Install python metis wrapper using pip:
```
pip install metis
```
5. Set METIS_DLL environment variable by adding the following script to ~/.bashrc:
```
export METIS_DLL=~/.local/lib/libmetis.so
```
6. Install dependencies using the following pip commands:
```
pip install nvidia-tensorflow
pip install networkx==1.11
pip install numpy==1.23.5
pip install scikit-learn==1.2.2
pip install scipy==1.10.1
pip install setuptools
```
## GNN AutoScale (GAS) Directory Setup
1. Here, user can reuse the same conda environment as the main CQSIGN directory as there are no conflicting dependencies with GNN AutoScale.
2. Install METIS from step 2, 3, 4, and 5 of Cluster-GCN Directory Setup if not done before.
3. Install remaining dependancies using pip:
```
pip install hydra-core==1.3.2
pip install setuptools
```
4. Go to CQSIGN/OtherBenchmarks/GNNAutoScale/pyg_autoscale and compile the C++ files using the following script:
```
python setup.py install
```

# Executing Benchmarks

## Preparing Dataset

## Pre-processing for ClusterGCN and GNN AutoScale

## Executing C-QSIGN, C-SIGN, and C-GCN

## Executing C-ClusterGCN

## Executing C-GAS (GNN AutoScale)

