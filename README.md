# Build and Train Vision Transformer from Scratch

![logo](.docs/logo.jpeg)

This is the supplementary repository for the tutorial

[Build and Train Vision Transformer from Scratch](https://medium.com/@michkravets/build-and-train-vision-transformer-from-scratch-f206c065bdf8)

## Install

After you've cloned the repository, you need to install required packages.

### Install for CPU / MPS

Run the following command to install packages:

```shell
pip install -r requirements.txt
```

### Install for CUDA

Install PyTorch libraries with the command from [official web-site](https://pytorch.org/get-started/locally/):

```shell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install the rest of the libraries with the command:

```shell
pip install -r requirements-cuda.txt
```
