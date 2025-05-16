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

# PRETRAINED WEIGHTS
You can find our pretrained weights [here!](https://drive.google.com/drive/folders/1F1ie30OzDnuEoMzrg0uaplnEvY3N2k2H?usp=sharing)

# RUNNING LITHOS CODE

Usage: main.py arguments

  ```
  -h, --help            show this help message and exit
  --lr LR               Initial learning rate (default: 1e-3)
  --epochs EPOCHS       Maximum number of epochs (default: 10)
  --batch BATCH         Batch size (default: 64)
  --fold FOLD           Fold to train (default: 1)
  --test                Only test the model
  --ft                  Fine-tune a model
  --resume              Resume training a model
  --model_type MODEL_TYPE
                        Model architecture (default: PolarViT)
  --name NAME           Name of the experiment (default: PolarViT_exp)
  --load_model LOAD_MODEL
                        Weights to load (default: best_acc)
  --n_classes N_CLASSES
                        Number of classes (default: 2)
  --root_dataset ROOT_DATASET
                        Root path to the dataset (default: LITHOS_DATASET_P/LITHOS_DATASET)
  --balance_dataloader  Balance the dataloader
  --balance_loss        Balance loss
  --use_alpha           Use learnable α-gate fusion (default)
  --use_linear          Use nn.Linear on concatenated
                        features
  --use_xpl             Use XPL data
  --optimizer {Adam,SGD}
                        Optimizer type (default: Adam)
  --momentum MOMENTUM   Momentum for SGD (default: 0.9)
  --scheduler {StepLR,CosineAnnealingLR,None}
                        LR scheduler type (default: StepLR)
  --step_size STEP_SIZE
                        Step size for StepLR scheduler (default: 10)
  --gamma GAMMA         Gamma for StepLR scheduler (default: 0.1)
  --T_max T_MAX         T_max for CosineAnnealingLR scheduler (default: 50)
```


