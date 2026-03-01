from sklearn.cluster import SpectralClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
class PerturbationSensitivityNetwork:
    """
    Task 6: Perturbation Sensitivity Network Analysis
    Identify gene modules and regulatory networks sensitive to drug perturbations
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def compute_gene_sensitivity(self, adata, test_dataset, n_perturbations=20):
        """
        Compute the sensitivity of each gene across different perturbations
        """
        print("Computing gene sensitivity across perturbations...")
        
        ptrb_col = 'sub_ptrb' if 'sub_ptrb' in adata.obs.columns else 'canonical_smiles'
        perturbations = adata.obs[ptrb_col].unique()[:n_perturbations]
        
        gene_names = list(adata.var_names)
        sensitivity_matrix = np.zeros((len(gene_names), len(perturbations)))
        
        for i, ptrb in enumerate(tqdm(perturbations, desc="Processing perturbations")):
            try:
                # Get control and perturbed data
                mask = adata.obs[ptrb_col] == ptrb
                ctrl_mask = mask & (adata.obs['condition'] == 'control')
                ptrb_mask = mask & (adata.obs['condition'] == 'perturb')
                
                if ctrl_mask.sum() == 0 or ptrb_mask.sum() == 0:
                    continue
                
                ctrl_data = adata[ctrl_mask]
                ptrb_data = adata[ptrb_mask]
                
                ctrl_exp = ctrl_data.X.A if hasattr(ctrl_data.X, 'A') else ctrl_data.X
                ptrb_exp = ptrb_data.X.A if hasattr(ptrb_data.X, 'A') else ptrb_data.X
                
                # Compute response magnitude (normalized log2FC)
                log2fc = np.log2((ptrb_exp.mean(axis=0) + 1) / (ctrl_exp.mean(axis=0) + 1))

                # Compute stability (1/CV)
                cv_ctrl = ctrl_exp.std(axis=0) / (ctrl_exp.mean(axis=0) + 1)
                cv_ptrb = ptrb_exp.std(axis=0) / (ptrb_exp.mean(axis=0) + 1)
                stability = 1 / (cv_ctrl + cv_ptrb + 0.1)

                # Sensitivity = response magnitude x stability
                sensitivity = np.abs(log2fc) * stability
                sensitivity_matrix[:, i] = sensitivity
                
            except Exception as e:
                continue
        
        return sensitivity_matrix, gene_names, perturbations
    
    def identify_sensitive_modules(self, sensitivity_matrix, gene_names, n_clusters=5):
        """
        Use clustering to identify co-responsive gene modules
        """
        print(f"Identifying {n_clusters} gene modules...")

        # Normalize
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        sensitivity_scaled = scaler.fit_transform(sensitivity_matrix)

        # Spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='nearest_neighbors',
            random_state=42
        )
        labels = clustering.fit_predict(sensitivity_scaled)
        
        # Organize results
        modules = {}
        for i in range(n_clusters):
            module_genes = [gene_names[j] for j in np.where(labels == i)[0]]
            module_sensitivity = sensitivity_matrix[labels == i].mean(axis=0)
            
            modules[f'Module_{i+1}'] = {
                'genes': module_genes,
                'n_genes': len(module_genes),
                'avg_sensitivity': module_sensitivity.mean(),
                'sensitivity_profile': module_sensitivity
            }
        
        return modules, labels
    
    def analyze_module_enrichment(self, modules):
        """
        Perform pathway enrichment analysis for each module
        """
        print("Performing pathway enrichment for modules...")
        
        module_enrichments = {}
        
        for module_name, module_data in modules.items():
            genes = module_data['genes'][:200]  # Limit the number of genes
            
            # STRING enrichment
            url = "https://string-db.org/api/json/enrichment"
            params = {
                "identifiers": "\r".join(genes),
                "species": 9606
            }
            
            try:
                response = requests.post(url, data=params)
                if response.ok:
                    enrichment = response.json()
                    
                    # Filter and organize results
                    filtered = []
                    for item in enrichment:
                        if item['category'] in ['KEGG', 'Reactome', 'GO Process']:
                            filtered.append({
                                'category': item['category'],
                                'term': item['term'],
                                'description': item['description'],
                                'fdr': item['fdr'],
                                'n_genes': item['number_of_genes']
                            })
                    
                    if filtered:
                        module_enrichments[module_name] = pd.DataFrame(filtered).nsmallest(10, 'fdr')
            except:
                continue
        
        return module_enrichments
    
    def build_sensitivity_network(self, sensitivity_matrix, gene_names, threshold=0.7):
        """
        Build a gene sensitivity co-response network
        """
        print("Building gene sensitivity co-response network...")

        # Compute inter-gene correlations
        from scipy.stats import spearmanr
        correlation_matrix, _ = spearmanr(sensitivity_matrix, axis=1)

        # Build network
        G = nx.Graph()
        n_genes = len(gene_names)
        
        for i in range(n_genes):
            for j in range(i+1, n_genes):
                if abs(correlation_matrix[i, j]) > threshold:
                    G.add_edge(
                        gene_names[i],
                        gene_names[j],
                        weight=abs(correlation_matrix[i, j])
                    )
        
        print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G, correlation_matrix
    
    def perform_complete_analysis(self, adata, test_dataset, n_perturbations=20,
                                 n_clusters=5, save_dir='sensitivity_analysis'):
        """
        Complete sensitivity analysis pipeline
        """
        os.makedirs(save_dir, exist_ok=True)

        # 1. Compute sensitivity matrix
        sensitivity_matrix, gene_names, perturbations = self.compute_gene_sensitivity(
            adata, test_dataset, n_perturbations
        )

        # 2. Identify modules
        modules, labels = self.identify_sensitive_modules(
            sensitivity_matrix, gene_names, n_clusters
        )

        # 3. Module enrichment analysis
        module_enrichments = self.analyze_module_enrichment(modules)

        # 4. Build network
        G, correlation_matrix = self.build_sensitivity_network(
            sensitivity_matrix, gene_names, threshold=0.7
        )

        # 5. Visualize
        self.visualize_results(
            sensitivity_matrix, gene_names, perturbations, modules, labels,
            G, module_enrichments, save_dir
        )
        
        return {
            'sensitivity_matrix': sensitivity_matrix,
            'modules': modules,
            'enrichments': module_enrichments,
            'network': G
        }
    
    def visualize_results(self, sensitivity_matrix, gene_names, perturbations,
                         modules, labels, G, enrichments, save_dir):
        """
        Comprehensive visualization
        """
        # 1. Sensitivity heatmap
        plt.figure(figsize=(14, 10))

        # Sort by module
        sorted_idx = np.argsort(labels)
        sorted_matrix = sensitivity_matrix[sorted_idx, :]
        
        sns.heatmap(sorted_matrix[:100], cmap='RdYlBu_r', center=0,
                   xticklabels=False, yticklabels=False,
                   cbar_kws={'label': 'Sensitivity Score'})
        plt.xlabel('Perturbations', fontsize=12)
        plt.ylabel('Genes (sorted by module)', fontsize=12)
        plt.title('Gene Sensitivity Landscape', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sensitivity_heatmap.png', dpi=300)
        plt.close()
        
        # 2. Module size distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        module_sizes = [m['n_genes'] for m in modules.values()]
        module_names = list(modules.keys())
        
        axes[0].bar(module_names, module_sizes, alpha=0.7)
        axes[0].set_xlabel('Module', fontsize=12)
        axes[0].set_ylabel('Number of Genes', fontsize=12)
        axes[0].set_title('Gene Module Sizes', fontsize=14)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 3. Module sensitivity
        module_sensitivities = [m['avg_sensitivity'] for m in modules.values()]
        
        axes[1].bar(module_names, module_sensitivities, alpha=0.7, color='orange')
        axes[1].set_xlabel('Module', fontsize=12)
        axes[1].set_ylabel('Average Sensitivity', fontsize=12)
        axes[1].set_title('Module Sensitivity Profiles', fontsize=14)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/module_analysis.png', dpi=300)
        plt.close()
        
        # 4. Network visualization (core only)
        if G.number_of_nodes() > 0:
            # Extract the largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G_core = G.subgraph(largest_cc).copy()

            if G_core.number_of_nodes() < 100:
                plt.figure(figsize=(12, 10))
                pos = nx.spring_layout(G_core, k=0.5, iterations=50)

                # Node color by module
                node_colors = [labels[gene_names.index(node)]
                             if node in gene_names else 0
                             for node in G_core.nodes()]
                
                nx.draw_networkx_nodes(G_core, pos, node_color=node_colors,
                                     cmap='tab10', node_size=100, alpha=0.7)
                nx.draw_networkx_edges(G_core, pos, alpha=0.2)
                
                plt.title('Gene Sensitivity Co-response Network', fontsize=14)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'{save_dir}/sensitivity_network.png', dpi=300)
                plt.close()
        
        print(f"Visualizations saved to {save_dir}/")
        
        # Save enrichment results
        if enrichments:
            with open(f'{save_dir}/module_enrichments.txt', 'w') as f:
                for module_name, enrich_df in enrichments.items():
                    f.write(f"\n{'='*60}\n{module_name}\n{'='*60}\n")
                    f.write(enrich_df.to_string())
                    f.write("\n")