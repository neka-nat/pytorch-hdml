# Hardness-Aware Deep Metric Learning

This is an unofficial implementation of ["Hardness-Aware Deep Metric Learning" (CVPR 2019 Oral)](https://arxiv.org/abs/1903.05503) in Pytorch.

## Installation

```
cd pytorch-hdml
pip install pipenv
pipenv install
```

## Download dataset

```
cd data
python cars196_downloader.py
python cars196_converter.py
```

## Train CARS196 dataset
Execute a training script. 
When executed, visdom is launched at the same time.

```
pipenv shell
./run.sh train_cars196.py
```

Access http://localhost:8097 in the browser.

## Reference

Official tensorflow implementation https://github.com/wzzheng/HDML