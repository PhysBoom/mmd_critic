# mmd-critic

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/yourusername/mmd-critic/blob/main/LICENSE)

A Python package for implementing the Maximum Mean Discrepancy Critic (MMD-Critic) method. This method is commonly used to find prototypes and criticisms (outliers, roughly speaking) in datasets.

## Installation

You can install the package via pip:

```bash
pip install mmd-critic
```

## Usage

```python
from mmd_critic import MMDCritic
from mmd_critic.kernels import RBFKernel

critic = MMDCritic(X, RBFKernel(sigma=1))

protos = critic.select_prototypes(50)
criticisms = critic.select_criticisms(10, protos)
```

See more in the [examples](https://github.com/PhysBoom/mmd_critic/tree/main/examples)

## More Info

Read my [article](https://medium.com/@physboom/the-mmd-critic-method-explained-c6a77f2dbf18) for more info on the MMD critic method. I also encourage you to read the original [paper](https://papers.nips.cc/paper_files/paper/2016/hash/5680522b8e2bb01943234bce7bf84534-Abstract.html).

## Acknowledgements

The implementation here is based on Been Kim's [original implementation](https://github.com/BeenKim/MMD-critic/tree/master) and [paper](https://papers.nips.cc/paper_files/paper/2016/hash/5680522b8e2bb01943234bce7bf84534-Abstract.html)