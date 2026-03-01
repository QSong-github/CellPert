import requests
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from tqdm import tqdm

class DiseaseGenePrioritization:
    """
    Task 5: Disease Gene Prioritization
    Identify disease-relevant key genes based on perturbation response and network topology
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def compute_gene_importance(self, control_exp, perturbed_exp, gene_names):
        """
        Compute gene importance scores - revised version
        Combines multiple metrics:
        1. Magnitude of expression change
        2. Consistency of change
        3. Statistical significance
        """
        print(f"Computing gene importance from {control_exp.shape[0]} control and {perturbed_exp.shape[0]} perturbed samples...")
        
        # Ensure inputs are numpy arrays
        if hasattr(control_exp, 'A'):
            control_exp = control_exp.A
        if hasattr(perturbed_exp, 'A'):
            perturbed_exp = perturbed_exp.A
        
        control_exp = np.asarray(control_exp, dtype=float)
        perturbed_exp = np.asarray(perturbed_exp, dtype=float)
        
        # 1. Compute log2 fold change
        ctrl_mean = control_exp.mean(axis=0)
        ptrb_mean = perturbed_exp.mean(axis=0)
        
        # Check data range to determine whether log transformation has already been applied
        if ctrl_mean.max() < 20:  # likely already log-transformed
            log2fc = ptrb_mean - ctrl_mean
        else:
            log2fc = np.log2(ptrb_mean + 1) - np.log2(ctrl_mean + 1)
        
        # 2. Compute change consistency (1 / coefficient of variation)
        # Use the reciprocal of standard deviation as a consistency measure
        ctrl_std = control_exp.std(axis=0)
        ptrb_std = perturbed_exp.std(axis=0)
        combined_std = (ctrl_std + ptrb_std) / 2
        consistency = 1 / (combined_std + 0.1)  # add 0.1 to avoid division by zero
        
        # 3. Compute statistical significance
        pvals = np.ones(len(gene_names))
        
        # t-test can only be performed when sample count > 1
        if control_exp.shape[0] > 1 and perturbed_exp.shape[0] > 1:
            try:
                # Perform t-test gene by gene
                for i in range(len(gene_names)):
                    ctrl_vals = control_exp[:, i]
                    ptrb_vals = perturbed_exp[:, i]
                    
                    # Check whether there is sufficient variance
                    if ctrl_vals.std() > 1e-8 or ptrb_vals.std() > 1e-8:
                        try:
                            _, p = ttest_ind(ctrl_vals, ptrb_vals, equal_var=False)
                            pvals[i] = p
                        except:
                            pvals[i] = 1.0
                    else:
                        pvals[i] = 1.0
                
                print(f"  Valid p-values: {np.sum(~np.isnan(pvals))}/{len(pvals)}")
                
            except Exception as e:
                print(f"  Warning: t-test failed ({e}), using fold change only")
                pvals = np.ones(len(gene_names))
        else:
            print(f"  Warning: Insufficient samples for t-test (need >1), using fold change only")
        
        # Handle NaN p-values
        pvals = np.nan_to_num(pvals, nan=1.0)
        
        # Compute -log10(p) as significance score
        # Add a small epsilon to avoid log(0)
        significance = -np.log10(pvals + 1e-300)
        # Clip extreme values
        significance = np.clip(significance, 0, 20)
        
        # 4. Combined score
        # Normalize each metric to the 0-1 range
        abs_log2fc = np.abs(log2fc)
        abs_log2fc_norm = (abs_log2fc - abs_log2fc.min()) / (abs_log2fc.max() - abs_log2fc.min() + 1e-8)
        
        consistency_norm = (consistency - consistency.min()) / (consistency.max() - consistency.min() + 1e-8)
        
        significance_norm = (significance - significance.min()) / (significance.max() - significance.min() + 1e-8)
        
        # Combined score: fold change has the highest weight
        importance_scores = (
            abs_log2fc_norm * 0.5 +      # fold change: 50%
            consistency_norm * 0.2 +      # consistency: 20%
            significance_norm * 0.3       # significance: 30%
        )
        
        # Re-normalize to 0-1
        importance_scores = (importance_scores - importance_scores.min()) / \
                          (importance_scores.max() - importance_scores.min() + 1e-8)
        
        gene_importance = pd.DataFrame({
            'gene': gene_names,
            'importance_score': importance_scores,
            'log2fc': log2fc,
            'consistency': consistency_norm,  # using normalized values
            'significance': significance_norm  # using normalized values
        }).sort_values('importance_score', ascending=False)
        
        print(f"  Top gene: {gene_importance.iloc[0]['gene']} (score: {gene_importance.iloc[0]['importance_score']:.4f})")
        print(f"  Score range: [{importance_scores.min():.4f}, {importance_scores.max():.4f}]")
        
        return gene_importance
    
    def integrate_string_network(self, important_genes, top_k=100, max_retries=3):
        """
        Integrate STRING network information and compute network centrality - improved version
        """
        print(f"\nIntegrating STRING network for top {min(top_k, len(important_genes))} genes...")
        
        # Limit the number of genes
        genes_to_query = important_genes[:min(top_k, len(important_genes))]
        
        # Retrieve STRING network
        url = "https://string-db.org/api/json/network"
        params = {
            "identifiers": "\r".join(genes_to_query),
            "species": 9606,
            "required_score": 400  # Lower threshold to capture more interactions
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, data=params, timeout=60)
                if response.ok:
                    network_data = response.json()
                    
                    if not network_data:
                        print("  No network data returned from STRING")
                        return None, None
                    
                    # Build graph
                    G = nx.Graph()
                    for edge in network_data:
                        G.add_edge(
                            edge['preferredName_A'],
                            edge['preferredName_B'],
                            score=edge['score']
                        )
                    
                    if G.number_of_nodes() == 0:
                        print("  Empty network")
                        return None, None
                    
                    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
                    
                    # Compute centrality
                    try:
                        centrality = nx.degree_centrality(G)
                        betweenness = nx.betweenness_centrality(G)
                        
                        network_scores = {}
                        for gene in G.nodes():
                            network_scores[gene] = centrality[gene] * 0.6 + betweenness[gene] * 0.4
                        
                        return network_scores, G
                    except Exception as e:
                        print(f"  Error computing centrality: {e}")
                        # Return at least degree centrality
                        centrality = nx.degree_centrality(G)
                        return dict(centrality), G
                
                else:
                    print(f"  STRING API error (attempt {attempt+1}/{max_retries}): {response.status_code}")
                    
            except Exception as e:
                print(f"  Failed to fetch STRING network (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
        
        print("  Failed to integrate STRING network after all retries")
        return None, None
    
    def prioritize_disease_genes(self, adata, disease_condition, test_dataset, 
                                cell_type=None, top_k=50):
        """
        Full disease gene prioritization analysis

        Args:
            adata: dataset
            disease_condition: disease-related perturbation condition
            test_dataset: test dataset
            cell_type: specific cell type
            top_k: number of top genes to return
        """
        print(f"\n{'='*70}")
        print(f"Prioritizing genes for disease condition: {disease_condition[:50]}")
        print(f"{'='*70}")
        
        # Identify perturbation identifier column
        ptrb_col = 'sub_ptrb' if 'sub_ptrb' in adata.obs.columns else 'canonical_smiles'
        
        # Filter data
        if cell_type:
            mask = (adata.obs[ptrb_col] == disease_condition) & \
                   (adata.obs['cell_type'] == cell_type)
        else:
            mask = adata.obs[ptrb_col] == disease_condition
        
        ctrl_mask = mask & (adata.obs['condition'] == 'control')
        ptrb_mask = mask & (adata.obs['condition'] == 'perturb')
        
        if ctrl_mask.sum() == 0 or ptrb_mask.sum() == 0:
            print(f"Insufficient data: {ctrl_mask.sum()} control, {ptrb_mask.sum()} perturbed")
            return None, None
        
        print(f"Using {ctrl_mask.sum()} control and {ptrb_mask.sum()} perturbed samples")
        
        # Get expression data
        ctrl_data = adata[ctrl_mask]
        ptrb_data = adata[ptrb_mask]

        ctrl_exp = ctrl_data.X.A if hasattr(ctrl_data.X, 'A') else ctrl_data.X
        ptrb_exp = ptrb_data.X.A if hasattr(ptrb_data.X, 'A') else ptrb_data.X
        gene_names = list(adata.var_names)

        # 1. Compute gene importance
        gene_importance = self.compute_gene_importance(ctrl_exp, ptrb_exp, gene_names)

        # 2. Integrate STRING network
        top_genes = gene_importance.head(100)['gene'].tolist()
        network_scores, G = self.integrate_string_network(top_genes)

        if network_scores:
            # Add network scores
            gene_importance['network_score'] = gene_importance['gene'].map(
                lambda x: network_scores.get(x, 0)
            )

            # Normalize network scores
            max_net_score = gene_importance['network_score'].max()
            if max_net_score > 0:
                gene_importance['network_score'] = gene_importance['network_score'] / max_net_score

            # Combined score: integrate importance and network
            gene_importance['final_score'] = (
                gene_importance['importance_score'] * 0.7 +
                gene_importance['network_score'] * 0.3
            )
        else:
            gene_importance['network_score'] = 0.0
            gene_importance['final_score'] = gene_importance['importance_score']

        # Sort
        gene_importance = gene_importance.sort_values('final_score', ascending=False)
        
        print(f"\n✓ Gene prioritization completed")
        print(f"Top 5 prioritized genes:")
        for idx, row in gene_importance.head(5).iterrows():
            print(f"  {row['gene']}: final_score={row['final_score']:.4f}, "
                  f"log2fc={row['log2fc']:.4f}, network={row['network_score']:.4f}")
        
        return gene_importance.head(top_k), G
    
    def visualize_gene_prioritization(self, gene_importance, G=None,
                                     top_k=20, save_path='gene_prioritization.png'):
        """
        Visualize gene prioritization results - improved version
        """
        if gene_importance is None or len(gene_importance) == 0:
            print("No gene importance data to visualize")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Gene score ranking
        top_genes = gene_importance.head(top_k)

        # Use final_score
        y_pos = np.arange(len(top_genes))
        axes[0].barh(y_pos, top_genes['final_score'], alpha=0.7, color='steelblue')
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(top_genes['gene'], fontsize=9)
        axes[0].set_xlabel('Priority Score', fontsize=12)
        axes[0].set_title(f'Top {top_k} Prioritized Disease Genes', fontsize=14, fontweight='bold')
        axes[0].grid(axis='x', alpha=0.3)
        axes[0].invert_yaxis()
        
        # 2. Score composition analysis
        top_10 = gene_importance.head(10)

        # Prepare stacked bar data
        importance_component = top_10['importance_score'].values * 0.7
        network_component = top_10['network_score'].values * 0.3
        
        x_pos = np.arange(len(top_10))
        axes[1].barh(x_pos, importance_component, alpha=0.7, 
                    label='Expression-based (70%)', color='coral')
        axes[1].barh(x_pos, network_component, left=importance_component,
                    alpha=0.7, label='Network-based (30%)', color='lightgreen')
        
        axes[1].set_yticks(x_pos)
        axes[1].set_yticklabels(top_10['gene'], fontsize=9)
        axes[1].set_xlabel('Score Contribution', fontsize=12)
        axes[1].set_title('Score Composition (Top 10 Genes)', fontsize=14, fontweight='bold')
        axes[1].legend(loc='lower right')
        axes[1].invert_yaxis()
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gene prioritization visualization saved to {save_path}")
        
        # If a network is available, also visualize it (optional)
        if G is not None and G.number_of_nodes() > 0 and G.number_of_nodes() <= 50:
            self._visualize_gene_network(G, top_genes['gene'].tolist()[:20], 
                                        save_path.replace('.png', '_network.png'))
    
    def _visualize_gene_network(self, G, hub_genes, save_path):
        """Visualize the gene network"""
        plt.figure(figsize=(12, 10))

        # Show only the subgraph related to hub genes
        hub_nodes = [n for n in G.nodes() if n in hub_genes]
        if len(hub_nodes) > 0:
            subG = G.subgraph(hub_nodes).copy()
        else:
            subG = G
        
        if subG.number_of_nodes() == 0:
            print("No network to visualize")
            return
        
        # Layout
        try:
            pos = nx.spring_layout(subG, k=1/np.sqrt(subG.number_of_nodes()),
                                  iterations=50, seed=42)
        except:
            pos = nx.circular_layout(subG)

        # Node size proportional to degree
        degrees = dict(subG.degree())
        node_sizes = [degrees[node] * 100 + 200 for node in subG.nodes()]

        # Draw
        nx.draw_networkx_edges(subG, pos, alpha=0.3, width=1)
        nx.draw_networkx_nodes(subG, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.8, edgecolors='black')
        nx.draw_networkx_labels(subG, pos, font_size=8, font_weight='bold')
        
        plt.title(f'Gene Interaction Network\n({subG.number_of_nodes()} genes, {subG.number_of_edges()} interactions)',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Gene network visualization saved to {save_path}")
    
    def batch_prioritize_drugs(self, adata, n_drugs=5, save_dir='./output/task5_gene_priority'):
        """
        Batch analysis of gene prioritization for multiple drugs
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        ptrb_col = 'sub_ptrb' if 'sub_ptrb' in adata.obs.columns else 'canonical_smiles'
        drugs = adata.obs[ptrb_col].unique()[:n_drugs]
        
        all_results = []
        
        for i, drug in enumerate(drugs):
            print(f"\n{'='*70}")
            print(f"Processing drug {i+1}/{len(drugs)}")
            
            gene_priority, G = self.prioritize_disease_genes(
                adata, drug, None, top_k=50
            )
            
            if gene_priority is not None:
                # Save results
                gene_priority.to_csv(f'{save_dir}/drug_{i+1}_gene_priority.csv', index=False)

                # Visualize
                self.visualize_gene_prioritization(
                    gene_priority, G,
                    save_path=f'{save_dir}/drug_{i+1}_visualization.png'
                )

                # Summarize
                all_results.append({
                    'drug_id': str(drug)[:50],
                    'n_prioritized_genes': len(gene_priority),
                    'top_gene': gene_priority.iloc[0]['gene'],
                    'top_score': gene_priority.iloc[0]['final_score'],
                    'network_available': G is not None
                })
        
        # Save summary
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(f'{save_dir}/summary.csv', index=False)
        
        print(f"\n✓ Batch analysis completed. Results saved to {save_dir}/")
        
        return summary_df