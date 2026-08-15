
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm
from torch_geometric.utils import to_dense_batch
import numpy as np
import pickle
import os
from collections import defaultdict
import gc
import pandas as pd
from torch_geometric.utils import to_dense_batch
from collections import defaultdict
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm
from torch_geometric.utils import to_dense_batch


class GINEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3, dropout=0.1):
        super(GINEncoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # LayerNorm in place of a pre-ReLU normalisation
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(BatchNorm(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Middle layers
        for i in range(1, num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(BatchNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            
            # Skip connection with proper initialization
            skip_conn = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(skip_conn.weight, gain=0.5)  # small initialisation
            self.skip_connections.append(skip_conn)
        
        # Last layer
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(BatchNorm(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Node-level latent projection layers with better initialization
        self.node_fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.node_fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # initialisation
        nn.init.xavier_uniform_(self.node_fc_mu.weight)
        nn.init.constant_(self.node_fc_mu.bias, 0)
        nn.init.xavier_uniform_(self.node_fc_logvar.weight)
        nn.init.constant_(self.node_fc_logvar.bias, -2)  # initialise to a small variance
        
        # learnable weight scale
        self.weight_scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, edge_index, batch):
        skip_outs = []
        
        # store the original expression values for weighting, with a numerical guard
        original_expr = x.squeeze(-1)  # [total_nodes]
        
        # First layer with residual scaling
        out = self.bns[0](self.convs[0](x, edge_index))
        out = F.relu(out)
        out = self.dropouts[0](out)
        skip_outs.append(out)
        
        # Middle layers with improved skip connections
        for i in range(1, self.num_layers - 1):
            identity = out  # keep the input for the residual
            out = self.bns[i](self.convs[i](out, edge_index))
            out = F.relu(out)
            out = self.dropouts[i](out)
            
            # skip connection, as a weighted combination
            if i > 0 and len(skip_outs) > 0:
                skip_proj = self.skip_connections[i-1](skip_outs[i-1])
                # learnable combination weights
                out = 0.7 * out + 0.3 * skip_proj + 0.1 * identity  # multiple connections
            
            skip_outs.append(out)
        
        # Last layer
        if self.num_layers > 1:
            identity = out
            out = self.bns[-1](self.convs[-1](out, edge_index))
            out = F.relu(out)
            out = self.dropouts[-1](out)
            
            # Add final skip connection with residual
            if len(self.skip_connections) > 0:
                skip_proj = self.skip_connections[-1](skip_outs[-1])
                out = 0.6 * out + 0.3 * skip_proj + 0.1 * identity
        
        # Node-level latent representations with constraint
        node_mu = self.node_fc_mu(out)
        node_logvar = torch.clamp(self.node_fc_logvar(out), -10, 10)  # bound logvar
        
        # graph-level representation
        # softmax-normalised weights, which are stabler
        expr_weights = torch.abs(original_expr) + 1e-6
        
        # graph-level representation per batch element
        unique_batch_indices = torch.unique(batch, sorted=True)
        graph_mu_list = []
        graph_logvar_list = []
        
        for batch_idx in unique_batch_indices:
            mask = (batch == batch_idx)
            
            curr_node_mu = node_mu[mask]
            curr_node_logvar = node_logvar[mask]
            curr_weights = expr_weights[mask]
            
            # softmax normalisation of the weights
            if len(curr_weights) > 1:
                # temperature controls how sharp the weight distribution is
                temperature = 0.5
                curr_weights_norm = F.softmax(curr_weights / temperature, dim=0)
            else:
                curr_weights_norm = torch.ones_like(curr_weights)
            
            curr_weights_norm = curr_weights_norm.unsqueeze(-1)
            
            # weighted mean, with a numerical guard
            graph_mu_i = torch.sum(curr_node_mu * curr_weights_norm, dim=0, keepdim=True)
            
            # log-sum-exp on logvar, for numerical stability
            weighted_logvar = curr_node_logvar + torch.log(curr_weights_norm + 1e-8)
            graph_logvar_i = torch.logsumexp(weighted_logvar, dim=0, keepdim=True) - \
                           torch.log(torch.tensor(curr_node_logvar.size(0), device=curr_node_logvar.device))
            
            graph_mu_list.append(graph_mu_i)
            graph_logvar_list.append(graph_logvar_i)
        
        graph_mu = torch.cat(graph_mu_list, dim=0)
        graph_logvar = torch.cat(graph_logvar_list, dim=0)
        
        # final numerical check
        graph_logvar = torch.clamp(graph_logvar, -10, 10)
        
        return node_mu, node_logvar, graph_mu, graph_logvar


class GINDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, num_layers=3, dropout=0.1):
        super(GINDecoder, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph-to-node projection: expand graph representation to all nodes
        self.graph_to_node_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize projection layers properly
        for layer in self.graph_to_node_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # First layer
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(BatchNorm(hidden_dim))
        self.dropouts.append(nn.Dropout(dropout))
        
        # Middle layers
        for i in range(1, num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(BatchNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            
            # Skip connection
            skip_conn = nn.Linear(hidden_dim, hidden_dim)
            nn.init.xavier_uniform_(skip_conn.weight, gain=0.5)
            self.skip_connections.append(skip_conn)
        
        # Output layer - no BatchNorm for final output
        if num_layers > 1:
            output_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout/2),  # Reduce dropout for output layer
                nn.Linear(hidden_dim, output_dim)
            )
            self.convs.append(GINConv(output_mlp))
            
            # Final skip projection (pre-defined to avoid dynamic creation)
            self.final_skip_proj = nn.Linear(hidden_dim, output_dim)
            nn.init.xavier_uniform_(self.final_skip_proj.weight, gain=0.3)
        else:
            # Single layer case - direct output
            output_mlp = nn.Sequential(
                nn.Linear(hidden_dim, output_dim)
            )
            self.convs.append(GINConv(output_mlp))
            self.final_skip_proj = None
        
        # Output scaling parameter
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, graph_z, edge_index, batch):
        """
        Decode graph-level latent representations to node-level features.
        
        Args:
            graph_z: Graph-level latent representations [num_graphs, latent_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes [total_nodes]
            
        Returns:
            out_dense: Reconstructed node features [num_graphs, max_nodes, output_dim]
            mask: Valid node mask [num_graphs, max_nodes]
        """
        device = graph_z.device
        num_graphs = graph_z.size(0)
        
        # Expand graph representations to node level
        # For each node, assign its corresponding graph's latent representation
        unique_batch_indices = torch.unique(batch, sorted=True)
        total_nodes = batch.size(0)
        
        # Create node features by expanding graph representations
        node_features = torch.zeros(total_nodes, self.hidden_dim, device=device)
        
        for i, batch_idx in enumerate(unique_batch_indices):
            # Find nodes belonging to this graph
            node_mask = (batch == batch_idx)
            num_nodes_in_graph = node_mask.sum()
            
            # Project graph representation to initial node features
            graph_feat = self.graph_to_node_proj(graph_z[i])  # [hidden_dim]
            
            # Assign same graph features to all nodes in this graph
            node_features[node_mask] = graph_feat.unsqueeze(0).expand(num_nodes_in_graph, -1)
        
        # Now process through GIN layers
        out = node_features
        skip_outs = []
        
        # First layer
        if self.num_layers > 1:
            out = self.bns[0](self.convs[0](out, edge_index))
            out = F.relu(out)
            out = self.dropouts[0](out)
            skip_outs.append(out)
            
            # Middle layers with improved skip connections
            for i in range(1, self.num_layers - 1):
                identity = out
                out = self.bns[i](self.convs[i](out, edge_index))
                out = F.relu(out)
                out = self.dropouts[i](out)
                
                # Skip connection
                if i > 0 and len(self.skip_connections) > i-1:
                    skip_proj = self.skip_connections[i-1](skip_outs[i-1])
                    out = 0.7 * out + 0.3 * skip_proj + 0.1 * identity
                skip_outs.append(out)
            
            # Last layer (output layer)
            identity = out
            out = self.convs[-1](out, edge_index)
            
            # Final skip connection with careful handling
            if len(skip_outs) > 0 and self.final_skip_proj is not None:
                last_skip = skip_outs[-1]
                skip_proj = self.final_skip_proj(last_skip)
                # More conservative skip connection weights for output
                out = 0.8 * out + 0.2 * skip_proj
        else:
            # Single layer case
            out = self.convs[0](out, edge_index)
        
        # Apply output scaling
        out = out * torch.sigmoid(self.output_scale)  # Constrain output range
        
        # Convert to dense batch format
        out_dense, mask = to_dense_batch(out, batch)
        return out_dense, mask




class GINVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=3, verbose=True):
        super(GINVAE, self).__init__()
        self.encoder = GINEncoder(input_dim, hidden_dim, latent_dim, num_layers=num_layers)
        self.decoder = GINDecoder(latent_dim, hidden_dim, input_dim, num_layers=num_layers)
        
        # Cache for reference latents (separate control and perturb)
        self._ctrl_ref_cache = None
        self._ptrb_ref_cache = None
        # Displacement of each individual compound, keyed by Morgan fingerprint.
        # Only populated when the reference is processed with per_compound=True.
        self._delta_by_fp = {}
        
        # Verbosity control
        self.verbose = verbose
        
    def _print(self, message, force=False):
        """Helper that controls printing."""
        if self.verbose or force:
            print(message)
            
    def _tqdm_write(self, message, force=False):
        """Print through tqdm.write so the progress bar is not disturbed."""
        if self.verbose or force:
            tqdm.write(message)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Encode - get both node-level and graph-level representations
        node_mu, node_logvar, graph_mu, graph_logvar = self.encoder(x, edge_index, batch)
        
        # Reparameterize at graph level (this is what we use for reconstruction)
        graph_z = self.reparameterize(graph_mu, graph_logvar)
        
        # Decode using graph-level latent representations
        recon_x, mask = self.decoder(graph_z, edge_index, batch)
        
        return recon_x, mask, node_mu, node_logvar, graph_mu, graph_logvar
    

    def process_and_save_reference_dataset(self, x_ref_dataset, save_path, max_samples_per_batch=1000, 
                                        save_every_n_batches=100, use_chunked_saving=True):
        """
        Process reference dataset and save all latent representations with metadata.
        Fixed to work at graph level instead of node level.
        
        Args:
            x_ref_dataset: Reference dataset 
            save_path: Path to save the processed reference data
            max_samples_per_batch: Maximum samples to process in one batch
            save_every_n_batches: Save progress every N batches
            use_chunked_saving: If True, save data in chunks to avoid memory buildup
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._print(f"Processing and saving reference dataset with {len(x_ref_dataset)} samples...")
        self._print(f"Using batch size: {max_samples_per_batch}, saving every {save_every_n_batches} batches")
        
        device = next(self.parameters()).device
        
        # Initialize storage
        if use_chunked_saving:
            chunk_dir = save_path.replace('.pkl', '_chunks')
            os.makedirs(chunk_dir, exist_ok=True)
            chunk_files = []
            current_chunk_data = []
        else:
            all_latent_data = []
        
        summary_stats = defaultdict(lambda: defaultdict(int))
        processed_count = 0
        
        # Process data in smaller batches
        num_batches = (len(x_ref_dataset) + max_samples_per_batch - 1) // max_samples_per_batch
        
        with torch.no_grad():
            # progress bar over batches
            batch_progress = tqdm(range(num_batches), 
                                desc="Processing reference batches", 
                                disable=not self.verbose)
            
            for batch_idx in batch_progress:
                start_idx = batch_idx * max_samples_per_batch
                end_idx = min((batch_idx + 1) * max_samples_per_batch, len(x_ref_dataset))
                
                # update the progress bar
                batch_progress.set_postfix({
                    'samples': f"{start_idx}-{end_idx-1}",
                    'processed': processed_count
                })
                
                batch_latent_data = []
                
                # progress bar over samples, shown only in verbose mode
                sample_range = range(start_idx, end_idx)
                if self.verbose and len(sample_range) > 100:
                    sample_iter = tqdm(sample_range, 
                                     desc=f"Batch {batch_idx+1} samples", 
                                     leave=False)
                else:
                    sample_iter = sample_range
                
                for i in sample_iter:
                    try:
                        ref_data = x_ref_dataset[i]
                        
                        # Move data to device
                        x = ref_data.x.to(device)
                        edge_index = ref_data.edge_index.to(device) if ref_data.edge_index is not None else None
                        batch_tensor = torch.zeros(x.size(0), dtype=torch.long).to(device)
                        
                        # Encode the sample
                        node_mu, node_logvar, graph_mu, graph_logvar = self.encoder(x, edge_index, batch_tensor)
                        
                        # Get graph-level latent representations (not node-level)
                        graph_z = self.reparameterize(graph_mu, graph_logvar)  # [1, latent_dim]
                        
                        # Extract metadata
                        metadata = {
                            'sample_idx': i,
                            'condition': str(ref_data.condition) if hasattr(ref_data, 'condition') and ref_data.condition is not None else 'unknown',
                            'celltype': str(ref_data.celltype) if hasattr(ref_data, 'celltype') and ref_data.celltype is not None else 'unknown',
                            'main_ptrb': str(ref_data.main_ptrb) if hasattr(ref_data, 'main_ptrb') and ref_data.main_ptrb is not None else 'none',
                            'sub_ptrb': str(ref_data.sub_ptrb) if hasattr(ref_data, 'sub_ptrb') and ref_data.sub_ptrb is not None else 'none',
                            'morgan_fp': str(ref_data.morgan_fp) if hasattr(ref_data, 'morgan_fp') and ref_data.morgan_fp is not None else None,
                            'num_nodes': x.size(0),
                            'num_edges': edge_index.size(1) if edge_index is not None else 0
                        }
                        
                        # Update summary stats
                        summary_stats['condition'][metadata['condition']] += 1
                        summary_stats['celltype'][metadata['celltype']] += 1
                        summary_stats['main_ptrb'][metadata['main_ptrb']] += 1
                        summary_stats['sub_ptrb'][metadata['sub_ptrb']] += 1
                        
                        # Store graph-level data (not node-level)
                        sample_data = {
                            'graph_mu': graph_mu.cpu().numpy().astype(np.float32),  # [1, latent_dim]
                            'graph_logvar': graph_logvar.cpu().numpy().astype(np.float32),  # [1, latent_dim]
                            'graph_z': graph_z.cpu().numpy().astype(np.float32),  # [1, latent_dim]
                            'metadata': metadata
                        }
                        
                        if use_chunked_saving:
                            current_chunk_data.append(sample_data)
                        else:
                            batch_latent_data.append(sample_data)
                        
                        processed_count += 1
                        
                        # Clear GPU tensors immediately
                        del x, edge_index, batch_tensor, node_mu, node_logvar, graph_mu, graph_logvar, graph_z
                        
                    except Exception as e:
                        self._tqdm_write(f"Warning: Failed to process sample {i}: {e}")
                        continue
                
                # Add batch data to main storage (if not using chunked saving)
                if not use_chunked_saving:
                    all_latent_data.extend(batch_latent_data)
                
                # Periodic saving and memory cleanup
                if (batch_idx + 1) % save_every_n_batches == 0 or batch_idx == num_batches - 1:
                    if use_chunked_saving:
                        # Save current chunk
                        if current_chunk_data:
                            chunk_file = os.path.join(chunk_dir, f'chunk_{len(chunk_files)}.pkl')
                            with open(chunk_file, 'wb') as f:
                                pickle.dump(current_chunk_data, f)
                            chunk_files.append(os.path.basename(chunk_file))
                            self._tqdm_write(f"Saved chunk with {len(current_chunk_data)} samples")
                            current_chunk_data = []
                    
                    # Aggressive memory cleanup
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    # Print memory usage if available (only if verbose)
                    if self.verbose:
                        try:
                            import psutil
                            memory_info = psutil.virtual_memory()
                            batch_progress.set_postfix({
                                'samples': f"{start_idx}-{end_idx-1}",
                                'processed': processed_count,
                                'memory': f"{memory_info.percent:.1f}%"
                            })
                        except ImportError:
                            pass
        
        self._print(f"Processing completed. Total samples processed: {processed_count}")
        
        # Prepare final save data
        if use_chunked_saving:
            # Save chunk index and metadata
            save_data = {
                'use_chunked_saving': True,
                'chunk_dir': chunk_dir,
                'chunk_files': chunk_files,
                'summary_stats': dict(summary_stats),
                'total_samples': processed_count,
                'latent_dim': None,  # Will be determined when loading
                'processing_info': {
                    'model_state': 'processed_chunked',
                    'device_used': str(device),
                    'batches_processed': num_batches,
                    'chunks_created': len(chunk_files)
                }
            }
        else:
            save_data = {
                'use_chunked_saving': False,
                'latent_data': all_latent_data,
                'summary_stats': dict(summary_stats),
                'total_samples': processed_count,
                'latent_dim': all_latent_data[0]['graph_z'].shape[-1] if all_latent_data else None,
                'processing_info': {
                    'model_state': 'processed',
                    'device_used': str(device),
                    'batches_processed': num_batches
                }
            }
        
        # Save the metadata/index file
        try:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
            
            self._print(f"Reference data metadata saved to {save_path}")
            if use_chunked_saving:
                self._print(f"Data chunks saved in {chunk_dir}")
            self._print(f"Total samples processed: {processed_count}")
            
            # Print summary
            if self.verbose:
                self._print("\nSummary statistics:")
                for group_type, group_counts in summary_stats.items():
                    self._print(f"  {group_type}:")
                    for group_name, count in sorted(group_counts.items()):
                        self._print(f"    {group_name}: {count}")
            
            return True
            
        except Exception as e:
            self._print(f"Error saving reference data: {e}", force=True)
            return False

    def load_and_compute_bio_context(self, save_path, group_type=None, group_name=None, max_ptrb_samples=None,
                                     per_compound=False):
        """
        Load processed reference data and compute biological context based on group criteria.
        Returns both control and perturb references separately for the correct formula.
        
        Args:
            save_path: Path to load the processed reference data
            group_type: Type of grouping ('condition', 'celltype', 'main_ptrb', 'sub_ptrb', or None)
            group_name: Specific group name to filter by
            max_ptrb_samples: Maximum number of perturbed samples to use
            
        Returns:
            tuple: (latent_ctrl_ref, latent_ptrb_ref, computation_info)
                latent_ctrl_ref: Control reference tensor [1, latent_dim]
                latent_ptrb_ref: perturb reference tensor [1, latent_dim]
                computation_info: dict with computation details
        """
        
        # Load processed data metadata
        try:
            with open(save_path, 'rb') as f:
                saved_data = pickle.load(f)
            self._print(f"Loaded reference data metadata from {save_path}")
            self._print(f"Total samples available: {saved_data['total_samples']}")
        except Exception as e:
            self._print(f"Error loading reference data: {e}", force=True)
            return None, None
        
        summary_stats = saved_data['summary_stats']
        use_chunked_saving = saved_data.get('use_chunked_saving', False)
        
        # Print available groups if group_type is specified
        if self.verbose and group_type is not None and group_type in summary_stats:
            self._print(f"\nAvailable {group_type} groups:")
            for group_val, count in sorted(summary_stats[group_type].items()):
                self._print(f"  {group_val}: {count} samples")
        
        # Validate group selection
        if group_type is not None and group_name is not None:
            if group_type not in summary_stats:
                self._print(f"Error: Group type '{group_type}' not found in saved data", force=True)
                return None, None
            
            if str(group_name) not in summary_stats[group_type]:
                self._print(f"Error: Group '{group_name}' not found in {group_type}", force=True)
                self._print(f"Available options: {list(summary_stats[group_type].keys())}", force=True)
                return None, None
        
        # Load and filter data
        if use_chunked_saving:
            self._print("Loading data from chunks...")
            chunk_dir = saved_data['chunk_dir']
            chunk_files = saved_data['chunk_files']
            
            ctrl_samples = []
            ptrb_samples = []
            total_filtered = 0
            
            # progress bar over chunk loading
            chunk_progress = tqdm(chunk_files, 
                                desc="Loading chunks", 
                                disable=not self.verbose)
            
            for chunk_file in chunk_progress:
                chunk_path = os.path.join(chunk_dir, chunk_file)
                chunk_progress.set_postfix({'file': chunk_file[:20] + '...' if len(chunk_file) > 20 else chunk_file})
                
                try:
                    with open(chunk_path, 'rb') as f:
                        chunk_data = pickle.load(f)
                    
                    # Filter chunk data
                    for sample_data in chunk_data:
                        # Apply group filter
                        if group_type is not None and group_name is not None:
                            if sample_data['metadata'][group_type] != str(group_name):
                                continue
                        
                        total_filtered += 1
                        
                        # Separate by condition
                        condition = sample_data['metadata']['condition']
                        if condition == 'control':
                            ctrl_samples.append(sample_data)
                        elif condition == 'perturb':
                            ptrb_samples.append(sample_data)
                    
                except Exception as e:
                    self._tqdm_write(f"Warning: Failed to load chunk {chunk_file}: {e}")
                    continue
            
            self._print(f"Loaded and filtered {total_filtered} samples from chunks")
            
        else:
            self._print("Loading data from single file...")
            all_latent_data = saved_data['latent_data']
            
            # Filter data based on group criteria
            if group_type is not None and group_name is not None:
                filtered_data = []
                
                # progress bar over filtering
                data_iter = tqdm(all_latent_data, 
                               desc=f"Filtering by {group_type}={group_name}", 
                               disable=not self.verbose)
                
                for sample_data in data_iter:
                    if sample_data['metadata'][group_type] == str(group_name):
                        filtered_data.append(sample_data)
                        
                self._print(f"After group filtering ({group_type}='{group_name}'): {len(filtered_data)} samples")
            else:
                filtered_data = all_latent_data
                self._print("Using all available samples (no group filtering)")
            
            # Separate control and perturb samples
            ctrl_samples = []
            ptrb_samples = []
            
            for sample_data in filtered_data:
                condition = sample_data['metadata']['condition']
                if condition == 'control':
                    ctrl_samples.append(sample_data)
                elif condition == 'perturb':
                    ptrb_samples.append(sample_data)
        
        if len(ctrl_samples) == 0 or len(ptrb_samples) == 0:
            self._print(f"Insufficient data: {len(ctrl_samples)} control, {len(ptrb_samples)} perturb samples", force=True)
            return None, None
        
        # Apply max_ptrb_samples limit
        if max_ptrb_samples is not None and len(ptrb_samples) > max_ptrb_samples:
            ptrb_samples = random.sample(ptrb_samples, max_ptrb_samples)
            self._print(f"Randomly selected {max_ptrb_samples} perturb samples")
        
        self._print(f"Computing biological context from {len(ctrl_samples)} control and {len(ptrb_samples)} perturb samples")
        
        # Compute biological context using graph-level representations
        device = next(self.parameters()).device
        
        # Extract graph-level latent representations (graph level, not node level)
        self._print("Extracting graph-level latent representations...")
        
        # progress bar over the control samples
        ctrl_graph_z_list = []
        ctrl_progress = tqdm(enumerate(ctrl_samples), 
                           total=len(ctrl_samples),
                           desc="Processing control samples",
                           disable=not self.verbose)
        
        for i, sample in ctrl_progress:
            # Each sample's graph_z is [1, latent_dim], we squeeze to [latent_dim]
            ctrl_graph_z_list.append(torch.tensor(sample['graph_z'], dtype=torch.float32).squeeze(0).to(device))
        
        # progress bar over the perturbed samples
        ptrb_graph_z_list = []
        ptrb_progress = tqdm(enumerate(ptrb_samples), 
                           total=len(ptrb_samples),
                           desc="Processing perturb samples",
                           disable=not self.verbose)
        
        for i, sample in ptrb_progress:
            # Each sample's graph_z is [1, latent_dim], we squeeze to [latent_dim]
            ptrb_graph_z_list.append(torch.tensor(sample['graph_z'], dtype=torch.float32).squeeze(0).to(device))
        
        # Calculate mean representations in batches to manage memory
        self._print("Computing mean representations...")
        
        # Process control samples in batches
        ctrl_batch_size = min(1000, len(ctrl_graph_z_list))
        ctrl_means = []
        for i in range(0, len(ctrl_graph_z_list), ctrl_batch_size):
            batch_tensors = ctrl_graph_z_list[i:i+ctrl_batch_size]
            batch_mean = torch.mean(torch.stack(batch_tensors), dim=0)  # [latent_dim]
            ctrl_means.append(batch_mean)
        latent_ctrl_ref = torch.mean(torch.stack(ctrl_means), dim=0)  # [latent_dim]
        
        # Process perturb samples in batches
        ptrb_batch_size = min(1000, len(ptrb_graph_z_list))
        ptrb_means = []
        for i in range(0, len(ptrb_graph_z_list), ptrb_batch_size):
            batch_tensors = ptrb_graph_z_list[i:i+ptrb_batch_size]
            batch_mean = torch.mean(torch.stack(batch_tensors), dim=0)  # [latent_dim]
            ptrb_means.append(batch_mean)
        latent_ptrb_ref = torch.mean(torch.stack(ptrb_means), dim=0)  # [latent_dim]
        
        # FIXED: Return both control and perturb references separately
        latent_ctrl_ref = latent_ctrl_ref.unsqueeze(0)  # [1, latent_dim]
        latent_ptrb_ref = latent_ptrb_ref.unsqueeze(0)  # [1, latent_dim]
        
        # Cache both references for prediction
        self._ctrl_ref_cache = latent_ctrl_ref
        self._ptrb_ref_cache = latent_ptrb_ref

        # One displacement per compound, in addition to the shared one above. The two
        # differ only in how the reference means are grouped: the shared displacement
        # averages every control and every perturbed sample, while this averages within
        # one compound at a time. Compounds whose controls are absent fall back to the
        # shared control mean, and a compound the reference never saw is handled at
        # prediction time by falling back to the shared displacement.
        self._delta_by_fp = {}
        if per_compound:
            fp_of = lambda s: (s.get('metadata') or {}).get('morgan_fp')
            if all(fp_of(s) is None for s in ptrb_samples):
                self._print("per_compound was requested but the cached reference carries no "
                            "morgan_fp; reprocess with force_reprocess=True to enable it",
                            force=True)
            else:
                ctrl_by_fp = {}
                for s in ctrl_samples:
                    f = fp_of(s)
                    if f is not None:
                        ctrl_by_fp.setdefault(f, []).append(
                            torch.tensor(s['graph_z'], dtype=torch.float32).squeeze(0))
                ptrb_by_fp = {}
                for s in ptrb_samples:
                    f = fp_of(s)
                    if f is not None:
                        ptrb_by_fp.setdefault(f, []).append(
                            torch.tensor(s['graph_z'], dtype=torch.float32).squeeze(0))
                shared_ctrl = latent_ctrl_ref.squeeze(0).cpu()
                for f, rows in ptrb_by_fp.items():
                    c = (torch.mean(torch.stack(ctrl_by_fp[f]), dim=0)
                         if f in ctrl_by_fp else shared_ctrl)
                    self._delta_by_fp[f] = torch.mean(torch.stack(rows), dim=0) - c
                self._print("Per-compound displacements: %d compounds" % len(self._delta_by_fp))
        
        # Prepare return information
        computation_info = {
            'group_type': group_type,
            'group_name': group_name,
            'ctrl_samples_used': len(ctrl_samples),
            'ptrb_samples_used': len(ptrb_samples),
            'ctrl_ref_shape': latent_ctrl_ref.shape,
            'ptrb_ref_shape': latent_ptrb_ref.shape,
            'use_chunked_loading': use_chunked_saving,
            'ctrl_mean_norm': torch.norm(latent_ctrl_ref).item(),
            'ptrb_mean_norm': torch.norm(latent_ptrb_ref).item(),
            'bio_context_norm': torch.norm(latent_ptrb_ref - latent_ctrl_ref).item()
        }
        
        self._print(f"References computed successfully.")
        self._print(f"Control ref shape: {latent_ctrl_ref.shape}, norm: {computation_info['ctrl_mean_norm']:.4f}")
        self._print(f"perturb ref shape: {latent_ptrb_ref.shape}, norm: {computation_info['ptrb_mean_norm']:.4f}")
        self._print(f"Biological context norm: {computation_info['bio_context_norm']:.4f}")
        
        # Clean up memory
        del ctrl_graph_z_list, ptrb_graph_z_list, ctrl_means, ptrb_means
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        return latent_ctrl_ref, latent_ptrb_ref, computation_info

    # convenience wrapper for the whole reference-dataset pipeline
    def process_reference_dataset(self, x_ref_dataset, save_path=None, group_type=None, group_name=None, 
                                max_ptrb_samples=None, max_samples_per_batch=100, force_reprocess=False,
                                per_compound=False):
        """Process the reference dataset and compute the biological context.

        If save_path does not exist the reference is processed and written there first.

        Args:
            x_ref_dataset: Reference dataset
            save_path: Path to save/load processed data (if None, will use temporary processing)
            group_type, group_name, max_ptrb_samples: Same as load_and_compute_bio_context
            max_samples_per_batch: For processing efficiency
            force_reprocess: If True, reprocess even if save_path exists
            
        Returns:
            tuple: (latent_ctrl_ref, latent_ptrb_ref, computation_info)
        """
        
        # Determine save path
        if save_path is None:
            save_path = 'temp_reference_latents.pkl'
            self._print("Using temporary save path:", save_path)
        
        # Check if we need to process
        need_to_process = force_reprocess or not os.path.exists(save_path)
        
        if need_to_process:
            self._print("Processing reference dataset...")
            success = self.process_and_save_reference_dataset(
                x_ref_dataset, save_path, max_samples_per_batch
            )
            if not success:
                return None, None, None
        else:
            self._print(f"Loading existing processed data from {save_path}")
        
        # Compute biological context
        return self.load_and_compute_bio_context(
            save_path, group_type, group_name, max_ptrb_samples, per_compound=per_compound
        )
    
    def predict(self, data, latent_ctrl_ref=None, latent_ptrb_ref=None,
                displacement='global'):
        """
        Enhanced prediction method using the correct formula:
        latent_perturb_query = latent_control_query + (latent_perturb_ref - latent_control_ref)
        
        Args:
            data: Query data (single graph or batch)
            latent_ctrl_ref: Control reference [1, latent_dim] (if None, use cached)
            latent_ptrb_ref: perturb reference [1, latent_dim] (if None, use cached)
            displacement: 'global' applies one displacement to every cell, which is the
                released behaviour and follows scGen. 'per_compound' applies that
                compound's own displacement instead, falling back to the shared one for
                a compound the reference never saw. The reference must have been
                processed with per_compound=True for this to have any effect.
        """
        # Use provided references or cached references
        if latent_ctrl_ref is None:
            if self._ctrl_ref_cache is None:
                self._print("Warning: No control reference provided and no cached reference available", force=True)
                # Fallback to reconstruction without biological context
                with torch.no_grad():
                    node_mu, node_logvar, graph_mu, graph_logvar = self.encoder(data.x, data.edge_index, data.batch)
                    graph_z = self.reparameterize(graph_mu, graph_logvar)
                    return self.decoder(graph_z, data.edge_index, data.batch)
            latent_ctrl_ref = self._ctrl_ref_cache
            
        if latent_ptrb_ref is None:
            if self._ptrb_ref_cache is None:
                self._print("Warning: No perturb reference provided and no cached reference available", force=True)
                # Fallback to reconstruction
                with torch.no_grad():
                    node_mu, node_logvar, graph_mu, graph_logvar = self.encoder(data.x, data.edge_index, data.batch)
                    graph_z = self.reparameterize(graph_mu, graph_logvar)
                    return self.decoder(graph_z, data.edge_index, data.batch)
            latent_ptrb_ref = self._ptrb_ref_cache
        
        with torch.no_grad():
            # Encode query data - get graph level representations
            node_mu_query, node_logvar_query, graph_mu_query, graph_logvar_query = self.encoder(
                data.x, data.edge_index, data.batch
            )
            
            # Get query's control state (what it would be without perturbation)
            graph_z_ctrl_qry = self.reparameterize(graph_mu_query, graph_logvar_query)  # [num_graphs, latent_dim]
            
            # Apply the correct formula:
            # latent_perturb_query = latent_control_query + (latent_perturb_ref - latent_control_ref)
            num_graphs = graph_z_ctrl_qry.size(0)
            
            # Expand references to match number of graphs in query
            latent_ctrl_ref_expanded = latent_ctrl_ref.expand(num_graphs, -1)  # [num_graphs, latent_dim]
            latent_ptrb_ref_expanded = latent_ptrb_ref.expand(num_graphs, -1)  # [num_graphs, latent_dim]
            
            # Apply the biological perturbation effect
            shared = latent_ptrb_ref_expanded - latent_ctrl_ref_expanded  # [num_graphs, latent_dim]
            if displacement == 'per_compound' and self._delta_by_fp:
                fps = getattr(data, 'morgan_fp', None)
                if fps is None or (isinstance(fps, list) and all(f is None for f in fps)):
                    # Processed graph caches written before morgan_fp was carried through
                    # have no fingerprint, and silently falling back would look like the
                    # option had no effect. Say so instead.
                    if not getattr(self, '_warned_no_fp', False):
                        self._print("per_compound requested but the query graphs carry no "
                                    "morgan_fp; delete the processed/ directory next to the "
                                    "h5ad so the graphs are rebuilt, otherwise the shared "
                                    "displacement is used", force=True)
                        self._warned_no_fp = True
                    biological_context = shared
                else:
                    if isinstance(fps, str):
                        fps = [fps]
                    rows, missing = [], 0
                    for j, f in enumerate(fps):
                        d = self._delta_by_fp.get(str(f))
                        if d is None:
                            rows.append(shared[j]); missing += 1
                        else:
                            rows.append(d.to(shared.device))
                    biological_context = torch.stack(rows)
                    self._last_fallback_count = missing
            elif displacement == 'per_compound':
                self._print("per_compound requested but no per-compound displacements are "
                            "available; using the shared displacement", force=True)
                biological_context = shared
            else:
                biological_context = shared
            graph_z_ptrb_qry = graph_z_ctrl_qry + biological_context  # [num_graphs, latent_dim]
            
            # Decode predicted expression using perturbed graph representations
            predicted_exp, mask = self.decoder(graph_z_ptrb_qry, data.edge_index, data.batch)
            
            # if self.verbose:
            #     self._print(f"Prediction completed. Output shape: {predicted_exp.shape}")
            #     self._print(f"Applied biological context with norm: {torch.norm(biological_context).item():.4f}")
            
        return predicted_exp, mask
    
    def get_biological_context(self):
        """
        Get the biological context (perturbation effect) from cached references.
        
        Returns:
            biological_context: latent_ptrb_ref - latent_ctrl_ref [1, latent_dim]
        """
        if self._ctrl_ref_cache is None or self._ptrb_ref_cache is None:
            self._print("Error: No cached references available. Run process_reference_dataset first.", force=True)
            return None
        
        biological_context = self._ptrb_ref_cache - self._ctrl_ref_cache
        return biological_context
    
    def predict_with_bio_context(self, data, bio_context=None):
        """
        Convenience method for prediction using biological context directly.
        
        Args:
            data: Query data
            bio_context: Biological context [1, latent_dim]. If None, compute from cached references.
        """
        if bio_context is None:
            bio_context = self.get_biological_context()
            if bio_context is None:
                return None, None
        
        # Use the references to apply the correct formula
        return self.predict(data, self._ctrl_ref_cache, self._ptrb_ref_cache)

    def set_verbose(self, verbose):
        """Turn verbose output on or off."""
        self.verbose = verbose





def loss_function(recon_x, x, node_mu, node_logvar, graph_mu, graph_logvar, args):
    # Reconstruction loss (node level)
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    
    # Graph-level KL divergence 
    kl_raw = -0.005 * torch.sum(1 + graph_logvar - graph_mu.pow(2) - graph_logvar.exp())
    
    graph_KLD = args.graph_kl_weight * kl_raw 
    
    return MSE + graph_KLD, MSE, graph_KLD