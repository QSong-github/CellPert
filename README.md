# CellPert: Graph Variational Learning with Biological Priors Enables Cross-Dataset Perturbation Prediction

CellPert is a graph neural network–based variational autoencoder framework for predicting single-cell transcriptomic responses to chemical perturbations. By modeling gene–gene interaction networks with Graph Isomorphism Networks (GIN) and a VAE latent space, CellPert transfers perturbation effects from reference datasets to unseen cell states, enabling a suite of downstream biological analyses.

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Installation](#installation)
3. [Datasets & Model Weights](#datasets--model-weights)
4. [Quick Start](#quick-start)
5. [Tutorial](#tutorial)
6. [Citation](#citation)

---

## Model Architecture

CellPert is built on Graph Isomorphism Network Variational AutoEncoder:

```
Input: Single-cell gene expression + STRING PPI graph
       ↓
GINEncoder (GIN layers + LayerNorm + skip connections)
       ↓
Node-level & Graph-level latent space (μ, σ²)
       ↓
Reparameterization trick: z = μ + ε·σ
       ↓
Biological context transfer:
  z_perturb_query = z_ctrl_query + (z_perturb_ref − z_ctrl_ref)
       ↓
GINDecoder (graph-to-node projection + GIN layers)
       ↓
Output: Predicted perturbed gene expression
```

**Key components:**

| Component | Description |
|-----------|-------------|
| `GINEncoder` | Multi-layer GIN with LayerNorm, skip connections, and expression-weighted graph pooling |
| `GINDecoder` | Graph-level → node-level projection with GIN refinement layers |
| `GINVAE` | Full VAE combining encoder/decoder; supports reference dataset caching for efficient inference |
| `ChunkedGeneGraphDataset` | Memory-efficient dataset class for large h5ad files with chunked loading |

**Model hyperparameters (default):**
- `input_dim = 1` (single-cell expression per gene)
- `hidden_dim = 300`
- `latent_dim = 100`
- `num_layers = 2`

---

## Installation

### Requirements

- Linux (tested on CUDA 12.9)
- Python 3.11
- Conda

### Using the provided environment file (recommended)

```bash
# Clone the repository
https://github.com/QSong-github/CellPert.git
cd CellPert

# Create the conda environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate info
```


---

## Datasets & Model Weights

All datasets and the pre-trained model weights are hosted on HuggingFace (public):

**[https://huggingface.co/datasets/Mike2481/CellPert/tree/main](https://huggingface.co/datasets/Mike2481/CellPert/tree/main)**

### Available Files

| File | Type | Description |
|------|------|-------------|
| `best_gin_vae_model_node_level.pth` | Model weights | Pre-trained GINVAE checkpoint |
| `lincs.zip` | Dataset (zip) | LINCS dataset |
| `minitahoe.tar.gz` | Dataset (tar.gz) | Tahoe mini dataset — 14 plates |

### Step 1 — Download model weights

Download `best_gin_vae_model_node_level.pth` manually from the HuggingFace page above and place it under `src/`:

```
CellPert/
└── src/
    └── best_gin_vae_model_node_level.pth
```

Or use `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Mike2481/CellPert",
    filename="best_gin_vae_model_node_level.pth",
    repo_type="dataset",
    local_dir="./src",
)
```

### Step 2 — Download and extract datasets

Download the archives from the HuggingFace page, then extract them:

```bash
# After downloading lincs.zip and minitahoe.tar.gz to ./data/
unzip ./data/lincs.zip               -d ./data/
tar  -xzf ./data/minitahoe.tar.gz    -C ./data/
```

Or download and extract programmatically:

```python
import os, zipfile, tarfile
from huggingface_hub import hf_hub_download

os.makedirs('./data', exist_ok=True)

# LINCS dataset
lincs_zip = hf_hub_download(
    repo_id="Mike2481/CellPert",
    filename="lincs.zip",
    repo_type="dataset",
    local_dir="./data",
)
with zipfile.ZipFile(lincs_zip, 'r') as z:
    z.extractall('./data/')

# Tahoe dataset (14 plates, distributed as a tarball)
tahoe_tar = hf_hub_download(
    repo_id="Mike2481/CellPert",
    filename="minitahoe.tar.gz",
    repo_type="dataset",
    local_dir="./data",
)
with tarfile.open(tahoe_tar, 'r:gz') as t:
    t.extractall('./data/')
```

### Expected data layout after extraction

```
CellPert/
├── src/
│   └── best_gin_vae_model_node_level.pth
└── data/
    ├── lincs/merged_all_965_with_morgan.h5ad   # LINCS (from lincs.zip)
    └── minitahoe/
        ├── p1_with_morgan.h5ad
        ├── p2_with_morgan.h5ad
        ├── ...
        └── p14_with_morgan.h5ad
```

---

## Quick Start

```python
import torch
import scanpy as sc
import sys; sys.path.insert(0, './src')
from model import GINVAE
from dataset import ChunkedGeneGraphDataset

# 1. Load pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GINVAE(input_dim=1, hidden_dim=300, latent_dim=100, num_layers=2).to(device)
model.load_state_dict(torch.load('./src/best_gin_vae_model_node_level.pth', map_location=device))
model.eval()

# 2. Load dataset
lincs_path = './data/merged_all_965_with_morgan.h5ad'
lincs_adata = sc.read_h5ad(lincs_path)
lincs_dataset = ChunkedGeneGraphDataset([lincs_path], split='test')

# 3. Build reference biological context (cached after first run)
model.process_reference_dataset(
    x_ref_dataset=lincs_dataset,
    save_path='./output/reference_latents.pkl',
    force_reprocess=False
)

# 4. Predict perturbation response for a query sample
from torch_geometric.data import DataLoader

sample = lincs_dataset[0]
batch = DataLoader([sample], batch_size=1, shuffle=False)
data = next(iter(batch)).to(device)

predicted_expression, mask = model.predict(data)
print(f'Predicted expression shape: {predicted_expression.shape}')
```

---

## Tutorial

[`tutorial.ipynb`](tutorial.ipynb) walks through a full CellPert use case end-to-end: download the data + pretrained weights from HuggingFace, load the GINVAE, build LINCS reference latents, predict perturbation responses on a Tahoe plate, and evaluate with Pearson / Spearman / DEG-delta on paired control–perturb samples.

---

## Citation

If you use CellPert in your research, please cite:


---

## License

MIT License © Qianqian Song Lab
