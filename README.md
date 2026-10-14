# DISTRA


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

# Clone DISTRA
git clone https://github.com/skybluewhy/DISTRA.git
cd DISTRA
```

# Data
The data used in the paper were collected from public data sets: https://www.10xgenomics.com/datasets and https://www.covid19cellatlas.org/index.patient.html.

# Usage of DISTRA

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
