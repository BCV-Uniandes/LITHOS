# LITHOS: Large-scale Imaging and Thin Section Optical-polarization Set for Automated Petrography
LITHOS Official Benchmark Repository

LITHOS is the largest publicly available dataset and benchmark for automated petrography, containing over 211,000 high-resolution polarized light image patches and 105,000 expert-annotated mineral grains across 25 categories. This repository provides the dataset, code, and pretrained models, including a dual-encoder transformer baseline that leverages multiple polarization modalities for improved mineral classification. LITHOS aims to accelerate research and reproducibility in automated mineralogical analysis from thin section images.

![2pols_all_minerals_final-2-cropped_page-0001](https://github.com/user-attachments/assets/099ef0c7-517d-45c6-8795-2c500ca50c3f)

# Installation

To create the conda environment for LITHOS, run the following commands:

```bash
conda env create -f environment.yaml
conda activate LITHOS
```

# Dataset Download

The LITHOS Dataset is available via a private (for now) Kaggle URL. To download:

1. Register for a Kaggle account if you don't have one.
2. Access the private Kaggle dataset link.
3. Follow the instructions on Kaggle to download the data (we recommend using curl for efficient downloading).
