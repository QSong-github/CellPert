# scGVP: Single-Cell Graph Variational Perturbation Model

scGVP is a graph neural network–based variational autoencoder framework for predicting single-cell transcriptomic responses to chemical perturbations. By modeling gene–gene interaction networks with Graph Isomorphism Networks (GIN) and a VAE latent space, scGVP transfers perturbation effects from reference datasets to unseen cell states, enabling a suite of downstream biological analyses.

---

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Installation](#installation)
3. [Datasets & Model Weights](#datasets--model-weights)
4. [Quick Start](#quick-start)
5. [Downstream Tasks](#downstream-tasks)
6. [Project Structure](#project-structure)
7. [Citation](#citation)

---

## Model Architecture

scGVP is built around **GINVAE** (Graph Isomorphism Network Variational AutoEncoder):

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
git clone https://github.com/<your-org>/scGVP.git
cd scGVP

# Create the conda environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate info
```

The `environment.yml` file pins all dependencies to exact versions, including:

| Package | Version |
|---------|---------|
| Python | 3.11 |
| PyTorch | 2.9.0 (CUDA 12.9) |
| torch-geometric | 2.6.1 |
| scanpy | 1.11.4 |
| anndata | 0.12.1 |
| rdkit | 2023.9.5 |
| scikit-learn | 1.7.1 |
| umap-learn | 0.5.9 |
| networkx | 3.2.1 |
| statsmodels | 0.14.5 |
| huggingface-hub | 0.22.2 |

> **Note:** The environment uses Tsinghua University mirror channels. If you are outside China, you may remove the mirror lines from `environment.yml` and replace them with standard `conda-forge` and `pytorch` channels before creating the environment.

---

## Datasets & Model Weights

All datasets and the pre-trained model weights are hosted on HuggingFace:

**[https://huggingface.co/datasets/Mike2481/Dataset4scGVP/tree/main](https://huggingface.co/datasets/Mike2481/Dataset4scGVP/tree/main)**

### Available Files

| File | Type | Description |
|------|------|-------------|
| `best_gin_vae_model_node_level.pth` | Model weights | Pre-trained GINVAE checkpoint |
| `lincs.zip` | Dataset (zip) | LINCS dataset |
| `tahoe.zip` | Dataset (zip) | Tahoe mini dataset — 14 plates |

### Step 1 — Download model weights

Download `best_gin_vae_model_node_level.pth` manually from the HuggingFace page above and place it under `src/`:

```
scGVP/
└── src/
    └── best_gin_vae_model_node_level.pth
```

Or use `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Mike2481/Dataset4scGVP",
    filename="best_gin_vae_model_node_level.pth",
    repo_type="dataset",
    local_dir="./src"
)
```

### Step 2 — Download and extract datasets

Download the zip files from the HuggingFace page, then extract them:

```bash
# After downloading lincs.zip and tahoe.zip to ./data/
unzip ./data/lincs.zip  -d ./data/
unzip ./data/tahoe.zip  -d ./data/tahoe/
```

Or download and extract programmatically:

```python
from huggingface_hub import hf_hub_download
import zipfile, os

os.makedirs('./data/tahoe', exist_ok=True)

# LINCS dataset
lincs_zip = hf_hub_download(
    repo_id="Mike2481/Dataset4scGVP",
    filename="lincs.zip",
    repo_type="dataset",
    local_dir="./data"
)
with zipfile.ZipFile(lincs_zip, 'r') as z:
    z.extractall('./data/')

# Tahoe dataset
tahoe_zip = hf_hub_download(
    repo_id="Mike2481/Dataset4scGVP",
    filename="tahoe.zip",
    repo_type="dataset",
    local_dir="./data"
)
with zipfile.ZipFile(tahoe_zip, 'r') as z:
    z.extractall('./data/tahoe/')
```

### Expected data layout after extraction

```
scGVP/
├── src/
│   └── best_gin_vae_model_node_level.pth
└── data/
    ├── merged_all_965_with_morgan.h5ad   # LINCS (from lincs.zip)
    └── tahoe/
        ├── p1_with_morgan.h5ad
        ├── p2_with_morgan.h5ad
        ├── ...
        └── p14_with_morgan.h5ad
```

### Dataset Overview

**LINCS Dataset** (`merged_all_965_with_morgan.h5ad`)
- ~965 unique chemical perturbations (canonical SMILES)
- Multiple human cell lines
- Paired control / perturbed single-cell profiles
- Key metadata columns: `condition` (control/perturb), `canonical_smiles`, `cell_type`, `morgan_fp`

**Tahoe Dataset** (14 plates, `p1_with_morgan.h5ad` – `p14_with_morgan.h5ad`)
- ~363 unique chemical perturbations (SMILES stored in `sub_ptrb`)
- 14 experimental plates with partially overlapping drug libraries
- Paired control / perturbed single-cell profiles
- Key metadata columns: `condition`, `sub_ptrb`, `morgan_fp`

---

## Quick Start

```python
import torch
import scanpy as sc
from downstreamTask.model import GINVAE
from downstreamTask.dataset import ChunkedGeneGraphDataset

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

## Downstream Tasks

scGVP supports six downstream analysis tasks. Each task has a corresponding Python module and Jupyter notebook in `downstreamTask/`.

### Task 1: Cross-Dataset Consistency Analysis
**Script:** `task1.py` | **Notebook:** `task1.ipynb`

Validates model consistency across LINCS and Tahoe datasets using three strategies:
1. **SMILES Exact Matching** — correlates predictions for drugs with identical SMILES strings
2. **Morgan Fingerprint Similarity** — finds structurally similar drugs (Tanimoto ≥ 0.85)
3. **Latent Space Alignment** — measures batch effect reduction via KS statistic, Silhouette score, and centroid similarity

```python
from downstreamTask.task1 import CrossDatasetConsistencyAnalysis

task1 = CrossDatasetConsistencyAnalysis(model, device)
metrics = task1.comprehensive_analysis(
    lincs_adata, tahoe_adatas, lincs_dataset, tahoe_datasets,
    save_dir='./output/task1_cross_dataset'
)
```

---

### Task 2: Drug-Disease Gene Network Analysis
**Script:** `task2.py` | **Notebook:** `task2.ipynb`

Identifies drug target genes and pathway mechanisms by integrating model predictions with STRING PPI networks:
- **DEG identification** with adaptive threshold relaxation
- **PPI network construction** via STRING API
- **Hub gene detection** (composite centrality: degree + betweenness + closeness)
- **Pathway enrichment** (KEGG, Reactome, GO Process)
- **Network visualisation** in multiple layouts (Kamada-Kawai, Shell)

```python
from downstreamTask.task2 import DrugDiseaseGeneNetwork

task2 = DrugDiseaseGeneNetwork(model, device)
result  = task2.analyze_drug_perturbation(lincs_adata, drug_smiles)           # single drug
summary = task2.batch_analyze_drugs(lincs_adata, n_drugs=5,
                                    save_dir='./output/task2_drug_analysis')   # batch
```

**Outputs:** hub gene CSVs, pathway enrichment CSVs, PPI network PDFs

---

### Task 3: Drug Repurposing
**Script:** `task3.py` | **Notebook:** `task3.ipynb`

Discovers repurposing candidates by comparing drug perturbation signatures in latent space:
- **Signature extraction** — model-predicted perturbation delta (pred − control)
- **Cosine similarity matrix** for all drug pairs
- **Candidate ranking** by similarity to known disease drugs
- **UMAP visualisation** of the drug perturbation space

```python
from downstreamTask.task3 import DrugRepurposing

task3 = DrugRepurposing(model, device, output_dir='./output/task3_drug_rep')
drug_signatures, drug_metadata = task3.extract_drug_signatures(lincs_adata, lincs_dataset, max_drugs=50)
similarity_df = task3.compute_drug_similarity(drug_signatures)
candidates    = task3.predict_repurposing_candidates(known_disease_drugs, drug_signatures)
task3.visualize_drug_space(drug_signatures, drug_metadata, known_disease_drugs)
```

**Outputs:** SMILES-to-name mapping, similarity matrix CSV, repurposing candidates CSV, UMAP plot

---

### Task 4: Drug Synergy Prediction
**Script:** `task4.py` | **Notebook:** `task4.ipynb`

Predicts synergistic or antagonistic drug combinations using the **Bliss independence model**:

$$\text{Synergy Score} = \text{Observed} - \text{Expected}, \quad \text{Expected} = d_1 + d_2 - d_1 \cdot d_2$$

- Positive score → synergy; Negative score → antagonism
- High-throughput screening across all pairwise drug combinations

```python
from downstreamTask.task4 import DrugSynergyPrediction

task4 = DrugSynergyPrediction(model, device)
synergy_results = task4.screen_drug_combinations(lincs_adata, lincs_dataset, n_combinations=30)
task4.visualize_synergy_landscape(synergy_results)
```

**Outputs:** ranked synergy/antagonism table, synergy landscape bar chart

---

### Task 5: Disease Gene Prioritization
**Script:** `task5.py` | **Notebook:** `task5.ipynb`

Identifies key disease-associated genes by combining expression-based scoring with PPI network topology:

$$\text{Importance} = 0.5 \times |\log_2\text{FC}| + 0.2 \times \text{Consistency} + 0.3 \times \text{Significance}$$
$$\text{Final Score} = 0.7 \times \text{Importance} + 0.3 \times \text{Network Score}$$

```python
from downstreamTask.task5 import DiseaseGenePrioritization

task5 = DiseaseGenePrioritization(model, device)
gene_priority, gene_network = task5.prioritize_disease_genes(
    lincs_adata, disease_drug_smiles, lincs_dataset, top_k=50
)
task5.visualize_gene_prioritization(gene_priority, gene_network)
summary = task5.batch_prioritize_drugs(lincs_adata, n_drugs=5,
                                        save_dir='./output/task5_gene_priority')
```

**Outputs:** gene priority CSVs per drug, score decomposition plots, summary CSV

---

### Task 6: Perturbation Sensitivity Network
**Script:** `task6.py` | **Notebook:** `task6.ipynb`

Identifies gene modules with coordinated responses across multiple perturbations:
- **Sensitivity matrix** [n_genes × n_perturbations] = |log2FC| × stability
- **Spectral clustering** to detect co-regulated gene modules
- **Module pathway enrichment** (KEGG, Reactome, GO Process)
- **Gene co-response network** via Spearman correlation (threshold ≥ 0.7)

```python
from downstreamTask.task6 import PerturbationSensitivityNetwork

task6 = PerturbationSensitivityNetwork(model, device)
results = task6.perform_complete_analysis(
    lincs_adata, lincs_dataset,
    n_perturbations=20, n_clusters=5,
    save_dir='./output/task6_sensitivity'
)
```

**Outputs:** sensitivity heatmap, module analysis plots, co-response network, enrichment text

---

## Project Structure

```
scGVP/
├── environment.yml                       # Conda environment (Python 3.11, CUDA 12.9)
├── README.md
├── src/                                  # Training code & model weights
│   ├── main.py                           # Training entry point
│   ├── model.py                          # GINVAE model definition
│   ├── dataset.py                        # Dataset loader
│   ├── graphbuilder.py                   # Gene graph construction
│   ├── metrics.py                        # Training metrics
│   ├── utils.py                          # Utility functions
│   ├── run_all.sh                        # Training launcher script
│   └── best_gin_vae_model_node_level.pth # ← place downloaded weights here
├── data/                                 # ← place extracted datasets here
│   ├── merged_all_965_with_morgan.h5ad   #   (from lincs.zip)
│   └── tahoe/
│       ├── p1_with_morgan.h5ad           #   (from tahoe.zip)
│       └── ...
└── downstreamTask/                       # Inference & downstream analysis
    ├── model.py                          # GINVAE (inference copy)
    ├── dataset.py                        # ChunkedGeneGraphDataset
    ├── graphbuilder.py                   # Graph construction utilities
    ├── metrics.py                        # Evaluation metrics
    ├── utils.py                          # Utility functions
    ├── alltask.py                        # Master script (all 6 tasks)
    ├── task1.py / task1.ipynb            # Cross-dataset consistency
    ├── task2.py / task2.ipynb            # Drug-disease gene network
    ├── task3.py / task3.ipynb            # Drug repurposing
    ├── task4.py / task4.ipynb            # Drug synergy prediction
    ├── task5.py / task5.ipynb            # Disease gene prioritization
    └── task6.py / task6.ipynb            # Perturbation sensitivity network
```

---

## Citation

If you use scGVP in your research, please cite:

```bibtex
@article{scGVP2024,
  title   = {scGVP: Single-Cell Graph Variational Perturbation Model},
  author  = {},
  journal = {},
  year    = {2024}
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.
