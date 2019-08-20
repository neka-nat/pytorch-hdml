# Hardness-Aware Deep Metric Learning

This is an unofficial implementation of ["Hardness-Aware Deep Metric Learning" (CVPR 2019 Oral)](https://arxiv.org/abs/1903.05503) in Pytorch.

## Instrallation

```
cd pytorch-hdml
pip install pipenv
pipenv install
```

## Dataset

```
cd data
python cars196_downloader.py
python cars196_converter.py
```

## Train CARS196 dataset

```
pipenv shell
python train_cars196.py
```