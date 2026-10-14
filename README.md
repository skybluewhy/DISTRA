## DISTRA
**DISTRA**: **DI**Sentangled **TR**anscriptomic‑TCR **A**rchitecture — clonality‑aware multi‑space disentanglement and selective TCR fusion for dissecting T‑cell states from paired single‑cell TCR and transcriptome data.

DISTRA is an RNA‑centered variational deep‑learning framework for integrating paired scRNA‑seq and scTCR‑seq data. Instead of fusing RNA and TCR features into one shared latent space, DISTRA disentangles transcriptomic variation into four biologically interpretable subspaces: **batch‑associated**, **cell‑identity‑associated**, **clonality‑associated**, and **residual variation**. TCR sequence information is selectively fused *only to the clonality‑associated subspace*, avoiding unwarranted global cross‑modal assumptions.

## ✨ Key Capabilities

- 🧩 **Multi‑space disentanglement**: Decompose T‑cell transcriptome into independent latent factors guided by structured priors, auxiliary supervision and clonotype‑aware contrastive learning.
- 🔍 **Antigen‑relevant representation**: The clonality subspace recovers antigen‑related transcriptional and effector programs **without antigen labels**.
- 🔗 **Selective TCR‑RNA fusion**: Combine TCR‑BERT embeddings with clonality latent space to boost antigen‑specificity discrimination; model can also run in RNA‑only mode.
- 🧬 **Conditional generation**: Generate factor‑driven transcriptomic profiles to interpret biological signatures for each latent subspace.
- 🩺 **Clinical cohort support**: Works on real‑world infection datasets (e.g. SARS‑CoV‑2) to uncover disease‑relevant T‑cell states, including early interferon‑dominated and late clonally expanded cytotoxic CD8⁺ T‑cell states.

## 📋 Abstract Summary

Paired single‑cell RNA and TCR sequencing provides complementary information for T‑cell functional state, antigen specificity and clonal lineage. Most existing multimodal methods enforce a global shared latent space, entangling batch, cell type, clonal and residual signals. DISTRA addresses this limitation by disentangling transcriptional variation and selectively coupling TCR features to clonality subspace. Benchmarks on 10x antigen‑specific dataset and independent SARS‑CoV‑2 cohort demonstrate improved disentanglement quality, antigen discrimination and biological interpretability for infection, vaccination and cancer immunology research.


## 🚀 Quick Highlights

- Input: Scanpy `AnnData` object containing scRNA counts + clonotype annotations + TCR amino‑acid sequences.
- Output: Disentangled latent embeddings (`z_r`, `z_d`, `z1_y`, `z2_y`) and optional fused cross‑modal embedding `e`.
- Downstream usage: clustering, UMAP visualization, antigen‑specificity prediction, conditional gene‑expression generation, clinical T‑cell state dissection.


# Installation
Setup conda environment:
```
# Create environment
conda create -n DISTRA python=3.11 -y
conda activate DISTRA

# Instaill requirements
pip install pytorch
pip install scanpy
pip install awkward
pip install transformers==4.46.3

# Clone DISTRA
git clone https://github.com/skybluewhy/DISTRA.git
cd DISTRA
```

# Data
The data used in the paper were collected from public data sets: https://www.10xgenomics.com/datasets and https://www.covid19cellatlas.org/index.patient.html.


# Usage of DISTRA

Prepare the 10X dataset (https://drive.google.com/drive/folders/1qB2a3x7wiDl4azMXiUEIZvKvESdtRISr?usp=sharing) in <BASE_FOLDER>/data/10X/.

Training DISTRA:
```
python train_10X.py
```
Get embeddings of DISTRA:
```
python get_emb_10X.py
```

Training DISTRA by fusing TCR sequence information:
```
python train_10X_fused_tcr_info.py
```
Get embeddings of DISTRA:
```
python get_emb_10X_fused_tcr_info.py
```
