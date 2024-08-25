# mmd_critic

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yourusername/mmd-critic/blob/main/LICENSE)

A Python package for implementing the Maximum Mean Discrepancy Critic (MMD-Critic) method. This method is commonly used to find prototypes and criticisms (outliers, roughly speaking) in datasets.

## Installation

You can install the package via pip:

```bash
pip install mmd-critic
```

## Usage

```
from mmd_critic import MMDCritic
from mmd_critic.kernels import RBFKernel

critic = MMDCritic(X, RBFKernel(sigma=1))

protos = critic.select_prototypes(50)
criticisms = critic.select_criticisms(10, protos)
```

See more in the [examples](https://github.com/PhysBoom/mmd_critic/tree/main/examples)