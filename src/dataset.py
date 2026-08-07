import os
import torch
from torch_geometric.data import Data, DataLoader
import scanpy as sc
import numpy as np
from graphbuilder import build_gene_graph_from_h5ad
import torch
import pandas as pd
from torch.utils.data import Dataset
import random
# compatibility shim for PyTorch 2.6+
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr])
except (ImportError, AttributeError):
    # older PyTorch, or the class is absent
    pass

try:
    # add the other classes that may be needed
    import torch_geometric.data.data as tg_data
    safe_classes = []
    for attr_name in dir(tg_data):
        attr = getattr(tg_data, attr_name)
        if isinstance(attr, type) and attr_name.startswith('Data'):
            safe_classes.append(attr)
    if safe_classes:
        torch.serialization.add_safe_globals(safe_classes)
except:
    pass

class ChunkedGeneGraphDataset(Dataset):
    """Chunked gene-graph dataset, for data too large to hold in memory.

    Subclasses Dataset rather than InMemoryDataset and loads one chunk at a time.
    """
    def __init__(self, h5ad_paths, edge_indices=None, edge_attrs=None, split='train',
                 transform=None, pre_transform=None, 
                 auto_build_graph=True, species=None, required_score=None, 
                 normalize_log1p=None, string_weight_scale=None, 
                 corr_topk_missing=None, corr_metric=None, corr_min=None,
                 chunk_size=5000, max_memory_gb=80):
        """chunk_size: samples per chunk.

        max_memory_gb: ceiling on memory use, in GB.
        """
        super().__init__()
        
        # keep every constructor argument
        if isinstance(h5ad_paths, str):
            self.h5ad_paths = [h5ad_paths]
        else:
            self.h5ad_paths = h5ad_paths
        self.dataset_name = self.h5ad_paths[-1].split('/')[-1].replace('.h5ad', '')
        self.split = split
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.transform = transform
        self.pre_transform = pre_transform
        
        # remaining arguments, handled as before
        num_files = len(self.h5ad_paths)
        self.edge_indices = self._process_edge_param(edge_indices, num_files)
        self.edge_attrs = self._process_edge_param(edge_attrs, num_files)
        self.auto_build_graph = auto_build_graph
        
        # handle the graph arguments
        default_params = {
            'species': 9606, 'required_score': 700, 'normalize_log1p': False,
            'string_weight_scale': 1000.0, 'corr_topk_missing': 5,
            'corr_metric': "pearson", 'corr_min': None
        }
        
        self.graph_params_list = []
        for i in range(num_files):
            file_params = {}
            for param_name, default_value in default_params.items():
                param_value = locals()[param_name]
                if param_value is None:
                    file_params[param_name] = default_value
                elif isinstance(param_value, list):
                    if len(param_value) == num_files:
                        file_params[param_name] = param_value[i]
                    else:
                        file_params[param_name] = param_value[0] if len(param_value) == 1 else default_value
                else:
                    file_params[param_name] = param_value
            self.graph_params_list.append(file_params)
        
        # set up the directories
        self.root_dir = os.path.dirname(self.h5ad_paths[0])
        self.processed_dir = os.path.join(self.root_dir, f'{self.dataset_name}/processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # decide whether the data still has to be processed
        self.chunk_info_file = os.path.join(self.processed_dir, f'{split}_chunk_info.json')
        
        if os.path.exists(self.chunk_info_file):
            # load the chunk index
            import json
            with open(self.chunk_info_file, 'r') as f:
                self.chunk_info = json.load(f)
            self.total_samples = self.chunk_info['total_samples']
            self.num_chunks = self.chunk_info['num_chunks']
            print(f"Found {self.num_chunks} chunks with {self.total_samples} total samples")
        else:
            # the data has to be processed
            print("Processing data into chunks...")
            self._process_to_chunks()
    
    def _process_edge_param(self, param, num_files):
        """Normalise the edge arguments."""
        if param is None:
            return [None] * num_files
        elif isinstance(param, (torch.Tensor, type(None))):
            return [param] * num_files
        elif isinstance(param, list):
            if len(param) == num_files:
                return param
            elif len(param) == 1:
                return param * num_files
            else:
                raise ValueError(f"edge argument list has the wrong length")
        else:
            raise ValueError("edge argument is malformed")
    
    def _process_to_chunks(self):
        """Process the data and write it out in chunks."""
        import gc
        import json
        
        total_samples = 0
        chunk_files = []
        current_chunk_data = []
        current_chunk_size = 0
        chunk_idx = 0
        
        # iterate over the files
        for file_idx, h5ad_path in enumerate(self.h5ad_paths):
            print(f"Processing {h5ad_path} into chunks...")
            
            # take the edge information
            current_edge_index = self.edge_indices[file_idx]
            current_edge_attr = self.edge_attrs[file_idx]
            
            if current_edge_index is None and self.auto_build_graph:
                print(f"  - Building gene graph...")
                current_params = self.graph_params_list[file_idx]
                try:
                    data_graph, gene_order, edges_df = build_gene_graph_from_h5ad(
                        h5ad_path=h5ad_path, **current_params
                    )
                    current_edge_index = data_graph.edge_index
                    current_edge_attr = data_graph.edge_attr
                    del data_graph, gene_order, edges_df
                    gc.collect()
                except Exception as e:
                    print(f"  - Warning: Failed to build graph: {e}")
                    current_edge_index = None
                    current_edge_attr = None
            
            # read the matrix
            adata = sc.read_h5ad(h5ad_path)
            expr = adata.X
            if not isinstance(expr, np.ndarray):
                expr = expr.toarray()
            
            # read obs
            obs_condition = adata.obs['condition'].values if 'condition' in adata.obs.columns else None
            obs_celltype = adata.obs['cell_type'].values if 'cell_type' in adata.obs.columns else None
            obs_main_ptrb = adata.obs['main_ptrb'].values if 'main_ptrb' in adata.obs.columns else None
            obs_sub_ptrb = adata.obs['sub_ptrb'].values if 'sub_ptrb' in adata.obs.columns else None
            
            g_mean = torch.tensor(expr.mean(axis=0), dtype=torch.float).reshape(-1, 1)
            g_var = torch.tensor(expr.var(axis=0), dtype=torch.float).reshape(-1, 1)
            
        
            # build the samples
            num_samples = expr.shape[0]
            for i in range(num_samples):
                # single-cell expression values
                cell_expr = torch.tensor(expr[i], dtype=torch.float).unsqueeze(1)  # [num_genes, 1]
                # concatenated features: [cell_expr, g_mean, g_var] -> [num_genes, 3]
                # x = torch.cat([cell_expr, g_mean, g_var], dim=1)
                x = cell_expr  # node features are the single-cell expression values alone
                data = Data(x=x, edge_index=current_edge_index, edge_attr=current_edge_attr)
                
                # attach the obs fields
                data.condition = obs_condition[i] if obs_condition is not None else "unknown"
                data.celltype = obs_celltype[i] if obs_celltype is not None else "unknown"
                data.main_ptrb = obs_main_ptrb[i] if obs_main_ptrb is not None else "none"
                data.sub_ptrb = obs_sub_ptrb[i] if obs_sub_ptrb is not None else "none"
                
                if self.pre_transform:
                    data = self.pre_transform(data)
                
                current_chunk_data.append(data)
                current_chunk_size += 1
                total_samples += 1
                
                # flush the chunk if it is full
                if current_chunk_size >= self.chunk_size:
                    chunk_file = os.path.join(self.processed_dir, f'{self.split}_chunk_{chunk_idx}.pt')
                    torch.save(current_chunk_data, chunk_file)
                    chunk_files.append(f'{self.split}_chunk_{chunk_idx}.pt')
                    
                    print(f"  - Saved chunk {chunk_idx} with {current_chunk_size} samples")
                    
                    # release memory
                    current_chunk_data = []
                    current_chunk_size = 0
                    chunk_idx += 1
                    gc.collect()
            
            # release this file
            del adata, expr
            gc.collect()
        
        # write the final chunk
        if current_chunk_data:
            chunk_file = os.path.join(self.processed_dir, f'{self.split}_chunk_{chunk_idx}.pt')
            torch.save(current_chunk_data, chunk_file)
            chunk_files.append(f'{self.split}_chunk_{chunk_idx}.pt')
            print(f"  - Saved final chunk {chunk_idx} with {current_chunk_size} samples")
        
        # write the chunk index
        self.chunk_info = {
            'total_samples': total_samples,
            'num_chunks': len(chunk_files),
            'chunk_files': chunk_files,
            'chunk_size': self.chunk_size
        }
        
        with open(self.chunk_info_file, 'w') as f:
            json.dump(self.chunk_info, f)
        
        self.total_samples = total_samples
        self.num_chunks = len(chunk_files)
        print(f"Data processed into {self.num_chunks} chunks with {self.total_samples} total samples")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        """Load one sample on demand."""
        # find which chunk holds this sample
        chunk_idx = idx // self.chunk_size
        sample_idx_in_chunk = idx % self.chunk_size
        
        # load that chunk unless it is already loaded
        if not hasattr(self, '_current_chunk') or self._current_chunk_idx != chunk_idx:
            chunk_file = os.path.join(self.processed_dir, f'{self.split}_chunk_{chunk_idx}.pt')
            self._current_chunk = torch.load(chunk_file, weights_only=False)
            self._current_chunk_idx = chunk_idx
        
        # take the sample
        data = self._current_chunk[sample_idx_in_chunk]
        
        if self.transform:
            data = self.transform(data)
        
        return data




class ChunkAwareRandomSampler:
    """Shuffle while respecting chunk boundaries."""
    def __init__(self, dataset, chunk_size, shuffle_chunks=True, shuffle_within_chunk=False):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.num_chunks = (len(dataset) + chunk_size - 1) // chunk_size
    
    def __iter__(self):
        # order the chunks
        chunk_order = list(range(self.num_chunks))
        if self.shuffle_chunks:
            random.shuffle(chunk_order)
        
        indices = []
        for chunk_idx in chunk_order:
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, len(self.dataset))
            chunk_indices = list(range(start, end))
            
            # whether to shuffle inside a chunk
            if self.shuffle_within_chunk:
                random.shuffle(chunk_indices)
            
            indices.extend(chunk_indices)
        
        return iter(indices)
    
    def __len__(self):
        return len(self.dataset)

# example use:
def create_chunked_dataset(h5ad_paths, **kwargs):
    """Convenience constructor for the chunked dataset."""
    return ChunkedGeneGraphDataset(
        h5ad_paths=h5ad_paths,
        chunk_size=5000,  # 5000 samples per chunk
        max_memory_gb=80,  # 80 GB memory ceiling
        **kwargs
    )

