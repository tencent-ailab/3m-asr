## Introduction

FastMoE is a third-party project to support Mixture of Experts (MoE) model for Pytorch. More details can be seen at [fastmoe](https://github.com/laekov/fastmoe)

Our code is based on fastmoe v0.1.2 and we have made some modifications for our own use.

## Installation

### Prerequisites

Pytorch with CUDA and NCCL with P2P communication support are required.

We have tested the repository with `pytorch1.8+cuda10.2+nccl2.7.8` and `pytorch1.9+cuda11.1+nccl2.10.3`

### Installing

The extra NCCL developer package is needed, which has to be consistent with your PyTorch's NCCL version. You can inpected the version by running `torch.cuda.nccl.version()` and access the [download link of all NCCL version](https://developer.nvidia.com/nccl/nccl-legacy-downloads) to download. 

When the prerequisites are ready, you can run the installation with:

```shell
# if you are building this package for different kinds of cuda devices,
# remember to set the environment variable `TORCH_CUDA_ARCH_LIST`
USE_NCCL=1 python setup.py install
```
