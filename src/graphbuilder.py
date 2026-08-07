import os
import time
import math
import json
import numpy as np
import pandas as pd
import scanpy as sc
import requests
from typing import Dict, List, Tuple, Optional
import torch
from torch_geometric.data import Data


# ---------------------------
# Utils
# ---------------------------
def _log(msg):
    print(f"[GeneGraph] {msg}", flush=True)

def _batch_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]


# ---------------------------
# STRING API helpers
# Docs: https://string-db.org/help/api/
# ---------------------------
STRING_BASE = "https://string-db.org/api"
# common taxonomy ids: human 9606, mouse 10090, rat 10116
def string_map_ids(genes: List[str], species: int = 9606, sleep=0.34) -> Dict[str, str]:
    """Map user gene symbols onto STRING protein identifiers.

    Returns dict[user_gene] -> string_id.
    """
    endpoint = f"{STRING_BASE}/json/get_string_ids"
    mapping = {}
    # STRING asks callers to rate-limit, to avoid 429
    for chunk in _batch_list(genes, 1000):
        params = {
            "identifiers": "\r".join(chunk),
            "species": species,
            "limit": 1
        }
        try:
            r = requests.post(endpoint, data=params, timeout=60)
            r.raise_for_status()
            js = r.json()
            for item in js:
                # preferredName is the canonical name; stringId looks like 9606.ENSP000003... or 9606.ENSG...
                user = item.get("queryItem")
                sid  = item.get("stringId")
                if user and sid and user not in mapping:
                    mapping[user] = sid
        except Exception as e:
            _log(f"STRING get_string_ids failed on chunk ({len(chunk)}): {e}")
        time.sleep(sleep)
    return mapping


def string_get_network(
    string_ids: List[str],
    species: int = 9606,
    required_score: int = 700,
    sleep=0.34
) -> pd.DataFrame:
    """Fetch the interaction network from STRING as pairwise edges with a combined score.

    Returns a DataFrame with columns [stringId_A, stringId_B, combined_score].
    """
    endpoint = f"{STRING_BASE}/tsv/network"
    rows = []
    for chunk in _batch_list(string_ids, 500):
        params = {
            "identifiers": "%0d".join(chunk),
            "species": species,
            "required_score": required_score,  # 0-1000
        }
        try:
            r = requests.post(endpoint, data=params, timeout=120)
            r.raise_for_status()
            # returns TSV: preferredName_A preferredName_B score ... stringId_A stringId_B ...
            lines = r.text.strip().splitlines()
            header = lines[0].split("\t")
            col2idx = {c: i for i, c in enumerate(header)}
            for line in lines[1:]:
                parts = line.split("\t")
                try:
                    sA = parts[col2idx.get("stringId_A")]
                    sB = parts[col2idx.get("stringId_B")]
                    score = float(parts[col2idx.get("score")])
                    rows.append((sA, sB, score))
                except Exception:
                    continue
        except Exception as e:
            _log(f"STRING network failed on chunk ({len(chunk)}): {e}")
        time.sleep(sleep)
    if not rows:
        return pd.DataFrame(columns=["stringId_A", "stringId_B", "combined_score"])
    df = pd.DataFrame(rows, columns=["stringId_A", "stringId_B", "combined_score"])
    return df


# ---------------------------
# Correlation graph helpers
# ---------------------------
def build_corr_edges(
    X: np.ndarray,
    genes: List[str],
    topk: int = 5,
    corr_metric: str = "pearson",
    abs_corr: bool = True,
    min_corr: Optional[float] = None,
) -> pd.DataFrame:
    """Build correlation kNN edges between genes from a cells-by-genes matrix.

    Returns an edge table [gene_i, gene_j, weight].
      topk: each gene is joined to its topk most correlated genes
      corr_metric: 'pearson' or 'spearman'
      abs_corr: rank by the absolute correlation
      min_corr: optional threshold; if set, edges below it are dropped
    """
    _log(f"Compute gene-gene correlation ({corr_metric}) ...")
    # X: cells x genes  -> corr on gene axis
    if corr_metric == "spearman":
        # Pearson on ranks, which approximates Spearman
        ranks = np.apply_along_axis(lambda v: pd.Series(v).rank().values, 0, X)
        C = np.corrcoef(ranks, rowvar=False)
    else:
        # pearson
        C = np.corrcoef(X, rowvar=False)  # genes x genes

    G = len(genes)
    edges = []
    for i in range(G):
        corr_i = C[i].copy()
        corr_i[i] = -np.inf  # drop self-edges
        # take the ranking
        if abs_corr:
            order = np.argsort(-np.abs(corr_i))
        else:
            order = np.argsort(-corr_i)
        picked = []
        for j in order:
            if min_corr is not None:
                if abs_corr and abs(corr_i[j]) < min_corr:
                    continue
                if not abs_corr and corr_i[j] < min_corr:
                    continue
            picked.append(j)
            if len(picked) >= topk:
                break
        for j in picked:
            w = abs(corr_i[j]) if abs_corr else corr_i[j]
            edges.append((genes[i], genes[j], float(w)))
    df = pd.DataFrame(edges, columns=["gene_i", "gene_j", "weight"])
    # make undirected: deduplicate and keep the larger weight
    df2 = df.copy()
    df2[["a", "b"]] = np.sort(df2[["gene_i", "gene_j"]].values, axis=1)
    df2 = df2.groupby(["a", "b"], as_index=False)["weight"].max()
    df2.rename(columns={"a": "gene_i", "b": "gene_j"}, inplace=True)
    return df2


# ---------------------------
# Graph builder (STRING + corr fallback)
# ---------------------------
def build_gene_graph_from_h5ad(
    h5ad_path: str,
    species: int = 9606,
    required_score: int = 700,
    normalize_log1p: bool = True,
    string_weight_scale: float = 1000.0,
    corr_topk_missing: int = 5,
    corr_metric: str = "pearson",
    corr_min: Optional[float] = None,
) -> Tuple[Data, List[str], pd.DataFrame]:
    """Build the gene graph from an h5ad file, with genes as nodes.

    1) prior edges from STRING, filtered by the confidence threshold
    2) for genes STRING does not cover, correlation kNN edges from expression

    Returns:
      PyG Data(x, edge_index, edge_attr)
      gene_order (list[str]), the node order
      edges_df (DataFrame), the final edge table (gene_i, gene_j, weight, source)
    """
    _log(f"Reading h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    X = adata.X.A if hasattr(adata.X, "A") else adata.X
    X = np.asarray(X, dtype=float)  # cells x genes
    genes: List[str] = list(map(str, adata.var_names.tolist()))
    _log(f"Cells: {X.shape[0]}, Genes: {len(genes)}")

    if normalize_log1p:
        _log("Normalizing: per-cell libsize -> 1e4, then log1p")
        
        # require non-negative values
        if hasattr(adata.X, "A"):
            X_check = adata.X.A
        else:
            X_check = adata.X
        
        if np.any(X_check < 0):
            print("Warning: Found negative values in data before normalization")
            # either clip to zero or raise
            # adata.X = np.maximum(adata.X, 0)
        
        sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
        sc.pp.log1p(adata)
        X = adata.X.A if hasattr(adata.X, "A") else adata.X
        X = np.asarray(X, dtype=float)

    # ========== 1) STRING prior network ==========
    _log("Mapping genes to STRING IDs ...")
    gene2sid = string_map_ids(genes, species=species)
    covered_genes = [g for g in genes if g in gene2sid]
    _log(f"STRING covered genes: {len(covered_genes)} / {len(genes)}")

    edges_list = []
    source_list = []
    if covered_genes:
        sids = [gene2sid[g] for g in covered_genes]
        _log("Fetching STRING network edges ...")
        net_df = string_get_network(sids, species=species, required_score=required_score)
        if not net_df.empty:
            # map stringId back to gene symbol, using only the covered genes
            sid2gene = {gene2sid[g]: g for g in covered_genes}
            # keep only edges within our gene set
            net_df = net_df[
                net_df["stringId_A"].isin(sid2gene) & net_df["stringId_B"].isin(sid2gene)
            ].copy()
            if not net_df.empty:
                net_df["gene_i"] = net_df["stringId_A"].map(sid2gene)
                net_df["gene_j"] = net_df["stringId_B"].map(sid2gene)
                # rescale to 0-1
                net_df["weight"] = net_df["combined_score"].astype(float) / string_weight_scale
                net_df = net_df[["gene_i", "gene_j", "weight"]]
                net_df["source"] = "STRING"
                edges_list.append(net_df)
                source_list.append("STRING")

    # ========== 2) fill the gaps with correlation edges ==========
    missing_genes = [g for g in genes if g not in covered_genes]
    if len(missing_genes) > 0 and corr_topk_missing > 0:
        _log(f"Building correlation edges for {len(missing_genes)} missing genes (kNN over all genes, deduplicated, largest weight kept)")
        # run kNN over all genes first, which is more stable, then keep the edges that touch a missing gene
        corr_df_all = build_corr_edges(
            X=X,
            genes=genes,
            topk=corr_topk_missing,
            corr_metric=corr_metric,
            abs_corr=True,
            min_corr=corr_min,
        )
        # record the source
        corr_df_all["source"] = np.where(
            corr_df_all["gene_i"].isin(missing_genes) | corr_df_all["gene_j"].isin(missing_genes),
            "CORR_MISSING",
            "CORR"
        )
        edges_list.append(corr_df_all)
        source_list.append("CORR")

    # merge the edges and deduplicate as undirected
    if edges_list:
        edges_df = pd.concat(edges_list, ignore_index=True)
        # make undirected: sort the endpoints and keep the largest weight
        pair = np.sort(edges_df[["gene_i", "gene_j"]].values, axis=1)
        edges_df["a"] = pair[:, 0]
        edges_df["b"] = pair[:, 1]
        edges_df = edges_df.groupby(["a", "b"], as_index=False).agg(
            weight=("weight", "max"),
            source=("source", lambda s: ";".join(sorted(set(s))))
        ).rename(columns={"a": "gene_i", "b": "gene_j"})
    else:
        edges_df = pd.DataFrame(columns=["gene_i", "gene_j", "weight", "source"])

    # build the graph; node order follows genes, matching adata.var_names
    gene2idx = {g: i for i, g in enumerate(genes)}
    # keep the edges whose endpoints are in the node set
    edges_df = edges_df[
        edges_df["gene_i"].isin(gene2idx) & edges_df["gene_j"].isin(gene2idx)
    ].copy()

    if edges_df.empty:
        _log("WARNING: No edges constructed. Creating empty edge_index with self-loops optional.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float32)
    else:
        # to indices
        u = edges_df["gene_i"].map(gene2idx).values
        v = edges_df["gene_j"].map(gene2idx).values
        # undirected graph: add both directions
        edge_index = torch.tensor(np.vstack([np.r_[u, v], np.r_[v, u]]), dtype=torch.long)
        w = edges_df["weight"].astype(float).values
        edge_attr = torch.tensor(np.r_[w, w], dtype=torch.float32)

    # node features: one expression value per sample, or a per-sample summary such as mean and variance
    # the general choice used here is the per-gene mean and variance over all cells, a 2-dimensional feature that does not grow with the sample count
    g_mean = np.asarray(X.mean(axis=0)).reshape(-1, 1)
    g_var  = np.asarray(X.var(axis=0)).reshape(-1, 1)
    node_feat = np.concatenate([g_mean, g_var], axis=1)  # [G, 2]
    x = torch.tensor(node_feat, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # return the gene order and the edge table, for reuse and for auditing
    edges_df = edges_df[["gene_i", "gene_j", "weight", "source"]].reset_index(drop=True)
    return data, genes, edges_df




