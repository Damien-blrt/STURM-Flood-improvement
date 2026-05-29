# STURM-Flood Model Improvements

This repository contains code for improving deep learning models for flood extent mapping using the STURM-Flood dataset.
By Damien BALLERAT (2026)

## Overview

The objective of this project is to reproduce baseline results and develop improved models for flood segmentation using Sentinel-1 and Sentinel-2 satellite imagery.

This repository focuses on:
- Model implementation and training
- Experimental improvements over baseline models
- Evaluation and comparison of results

## Scope

The code provided here includes:
- Training pipelines for flood segmentation models
- Model architectures and modifications
- Evaluation scripts and metrics computation
- Inference utilities

This repository does **not** host the dataset or pretrained models.

## Dataset

The dataset used in this project is available on Zenodo:

- Sentinel-1 and Sentinel-2 tiles (128×128, 10 m resolution)
- Pixel-wise flood masks
- 60 flood events worldwide

Access the dataset: https://doi.org/10.5281/zenodo.12748982

## Relation to Original Work

This project builds upon the original STURM-Flood work:
- Baseline models and methodology are reproduced
- Additional experiments are conducted to improve performance

Original repository: https://github.com/STURM-WEO/STURM-Flood

Reference paper:  
*STURM-Flood: a curated dataset for deep learning-based flood extent mapping leveraging Sentinel-1 and Sentinel-2 imagery* (2025)

## Usage

### 1. Setup

build the image:
```bash
docker build -t [Name image] .
```
run the container: 
```bash
docker run --rm --gpus all -it \
-p 8888:8888 \
-u $(id -u):$(id -g) \
-v ~/[local path to repo]:/workspace \
[Name image] \
jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

After launching, a URL like the following will be generated: 
http://127.0.0.1:8888/?token=XXXX

Open this link in your browser to access the notebook.

All experiments were executed inside this container to ensure reproducibility.
if you want to run the training run scripts, please go inside the container.

## Metrics

Models are evaluated using standard segmentation metrics:

Intersection over Union (IoU)
F1-score
Dice coefficient

## Notes
This repository is focused on experimentation and model improvement
Results may vary depending on training setup and hyperparameters
Users are expected to download and prepare the dataset separately

## Citation

If you use this repository, please also cite the original paper:

Notarangelo, N. M., Wirion, C., & van Winsen, F. (2025).
STURM-Flood: A curated dataset for deep learning-based flood extent mapping leveraging Sentinel-1 and Sentinel-2 imagery.
Big Earth Data. https://doi.org/10.1080/20964471.2025.2458714

## License

Same as the original dataset and associated resources.