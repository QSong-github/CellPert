import requests
import networkx as nx
from scipy.stats import hypergeom
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
# Set font parameters to avoid font warnings
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype']  = 42
mpl.rcParams['svg.fonttype'] = 'none'

# Try to use Times font; fall back to system default serif font if unavailable
try:
    import matplotlib.font_manager as fm
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    if 'Times New Roman' in available_fonts:
        mpl.rcParams['font.family'] = 'Times New Roman'
    elif 'DejaVu Serif' in available_fonts:
        mpl.rcParams['font.family'] = 'DejaVu Serif'
    else:
        mpl.rcParams['font.family'] = 'serif'
        print("Note: Using system default serif font")
except:
    mpl.rcParams['font.family'] = 'sans-serif'
    print("Note: Using system default sans-serif font")
# Scientific color scheme - mixed Set2 and Paired palettes

# ========== Added: SMILES conversion utility class ==========
class SMILESConverter:
    """
    SMILES conversion utility class.
    Uses RDKit to convert SMILES strings to drug names / molecular formulas.
    """
    
    def __init__(self):
        self.conversion_cache = {}  # Cache for conversion results
        
    def smiles_to_name(self, smiles):
        """
        Convert a SMILES string to a human-readable drug name.
        Priority: IUPAC name > molecular formula > simplified SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Drug name string
        """
        if smiles in self.conversion_cache:
            return self.conversion_cache[smiles]
        
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                name = smiles[:30] + "..." if len(smiles) > 30 else smiles
                self.conversion_cache[smiles] = name
                return name
            
            # Get molecular formula
            formula = rdMolDescriptors.CalcMolFormula(mol)

            # Get molecular weight
            mw = rdMolDescriptors.CalcExactMolWt(mol)

            # Combine name: molecular formula (MW: xxx)
            name = f"{formula} (MW: {mw:.1f})"
            
            self.conversion_cache[smiles] = name
            return name
            
        except Exception as e:
            print(f"SMILES conversion error for {smiles[:30]}: {e}")
            name = smiles[:30] + "..." if len(smiles) > 30 else smiles
            self.conversion_cache[smiles] = name
            return name
    
    def batch_convert(self, smiles_list):
        """
        Batch-convert a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            DataFrame with conversion results
        """
        results = []
        for smiles in smiles_list:
            name = self.smiles_to_name(smiles)
            
            try:
                from rdkit import Chem
                from rdkit.Chem import rdMolDescriptors
                
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    formula = rdMolDescriptors.CalcMolFormula(mol)
                    mw = rdMolDescriptors.CalcExactMolWt(mol)
                else:
                    formula = "N/A"
                    mw = 0
            except:
                formula = "N/A"
                mw = 0
            
            results.append({
                'smiles': smiles,
                'name': name,
                'formula': formula,
                'molecular_weight': mw
            })
        
        return pd.DataFrame(results)
    
    def save_conversion_table(self, smiles_list, save_path='smiles_conversion.csv'):
        """
        Save the SMILES conversion table to a CSV file.

        Args:
            smiles_list: List of SMILES strings
            save_path: Output file path
        """
        df = self.batch_convert(smiles_list)
        df.to_csv(save_path, index=False)
        print(f"SMILES conversion table saved to {save_path}")
        return df


class DrugDiseaseGeneNetwork:
    """
    Task 2: Drug-disease gene network analysis
    Combines STRING PPI network with model predictions to identify core pathways of drug action
    """

    def __init__(self, model, device, species=9606):
        self.model = model
        self.device = device
        self.species = species
        self.string_base = "https://string-db.org/api"
        self.smiles_converter = SMILESConverter()  # Added: initialize SMILES converter
        
    def get_deg_genes(self, control_exp, perturbed_exp, gene_names,
                     log2fc_threshold=0.5, padj_threshold=0.1, min_genes=10):
        """
        Identify differentially expressed genes (DEGs) - improved version

        Args:
            control_exp: Control expression matrix [n_samples, n_genes]
            perturbed_exp: Perturbed expression matrix [n_samples, n_genes]
            gene_names: List of gene names
            log2fc_threshold: Log2 fold change threshold (lowered threshold)
            padj_threshold: Adjusted p-value threshold (relaxed threshold)
            min_genes: Minimum number of DEGs to return
        """
        print(f"Computing DEGs from {control_exp.shape[0]} control and {perturbed_exp.shape[0]} perturbed samples...")
        
        # Ensure inputs are numpy arrays
        if hasattr(control_exp, 'A'):
            control_exp = control_exp.A
        if hasattr(perturbed_exp, 'A'):
            perturbed_exp = perturbed_exp.A

        control_exp = np.asarray(control_exp, dtype=float)
        perturbed_exp = np.asarray(perturbed_exp, dtype=float)

        # Compute mean expression
        ctrl_mean = control_exp.mean(axis=0)
        ptrb_mean = perturbed_exp.mean(axis=0)

        # Method 1: if data is already in log space, use direct difference
        # Check data range to determine whether log transformation has been applied
        if ctrl_mean.max() < 20:  # likely already log-transformed
            print("Data appears to be log-transformed, using direct difference")
            log2fc = ptrb_mean - ctrl_mean
        else:
            # Raw count data, log transformation required
            print("Data appears to be raw counts, applying log2 transformation")
            log2fc = np.log2(ptrb_mean + 1) - np.log2(ctrl_mean + 1)
        
        # Compute p-values
        pvals = np.ones(len(gene_names))

        # Perform t-test only for genes showing variation
        try:
            if control_exp.shape[0] > 1 and perturbed_exp.shape[0] > 1:
                _, pvals = ttest_ind(control_exp, perturbed_exp, axis=0, equal_var=False)
                # Handle NaN values
                pvals = np.nan_to_num(pvals, nan=1.0)
            else:
                print("Warning: Insufficient samples for t-test, using fold change only")
        except Exception as e:
            print(f"Warning: t-test failed ({e}), using fold change only")
        
        # FDR correction
        try:
            _, padj, _, _ = multipletests(pvals, method='fdr_bh')
        except Exception as e:
            print(f"Warning: FDR correction failed ({e}), using raw p-values")
            padj = pvals

        # Filter DEGs - use progressively relaxed criteria
        deg_mask = (np.abs(log2fc) > log2fc_threshold) & (padj < padj_threshold)

        # If too few DEGs found, progressively relax criteria
        if deg_mask.sum() < min_genes:
            print(f"Only {deg_mask.sum()} DEGs found with strict criteria, relaxing thresholds...")

            # Strategy 1: relax p-value threshold
            deg_mask = (np.abs(log2fc) > log2fc_threshold) & (padj < 0.2)

            if deg_mask.sum() < min_genes:
                # Strategy 2: further relax p-value and log2fc thresholds
                deg_mask = (np.abs(log2fc) > 0.25) & (padj < 0.3)

            if deg_mask.sum() < min_genes:
                # Strategy 3: use fold change only, select top-changing genes
                print(f"Still only {deg_mask.sum()} DEGs, selecting top {min_genes} by fold change")
                top_indices = np.argsort(np.abs(log2fc))[-min_genes:]
                deg_mask = np.zeros(len(gene_names), dtype=bool)
                deg_mask[top_indices] = True
        
        deg_genes = [gene_names[i] for i in np.where(deg_mask)[0]]
        deg_log2fc = log2fc[deg_mask]
        deg_padj = padj[deg_mask]
        
        print(f"Identified {len(deg_genes)} DEGs")
        if len(deg_genes) > 0:
            print(f"  Log2FC range: [{deg_log2fc.min():.3f}, {deg_log2fc.max():.3f}]")
            print(f"  Adj. p-value range: [{deg_padj.min():.3e}, {deg_padj.max():.3e}]")
        
        return deg_genes, deg_log2fc, deg_padj
    
    def query_string_network(self, genes, required_score=400, max_retries=3):
        """
        Retrieve gene interaction network from the STRING database - improved version

        Args:
            genes: List of gene symbols
            required_score: Minimum interaction score (threshold lowered from 700 to 400)
            max_retries: Maximum number of retry attempts
        """
        if len(genes) == 0:
            print("No genes provided for STRING query")
            return None, None
        
        print(f"Querying STRING database for {len(genes)} genes...")
        
        # Step 1: Map genes to STRING IDs
        url = f"{self.string_base}/json/get_string_ids"
        params = {
            "identifiers": "\r".join(genes),
            "species": self.species,
            "limit": 1
        }
        
        mapping = {}
        for attempt in range(max_retries):
            try:
                response = requests.post(url, data=params, timeout=30)
                if response.ok:
                    result = response.json()
                    for item in result:
                        query_gene = item.get('queryItem', '')
                        string_id = item.get('stringId', '')
                        if query_gene and string_id:
                            mapping[query_gene] = string_id
                    break
                else:
                    print(f"STRING ID mapping failed (attempt {attempt+1}/{max_retries}): {response.status_code}")
            except Exception as e:
                print(f"STRING ID mapping error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
        
        if not mapping:
            print("Failed to map any genes to STRING IDs")
            return None, None
        
        print(f"Mapped {len(mapping)}/{len(genes)} genes to STRING IDs")
        
        # Step 2: Retrieve interaction network
        url = f"{self.string_base}/tsv/network"
        params = {
            "identifiers": "%0d".join(mapping.values()),
            "species": self.species,
            "required_score": required_score
        }
        
        edges = []
        for attempt in range(max_retries):
            try:
                response = requests.post(url, data=params, timeout=60)
                if response.ok:
                    lines = response.text.strip().split('\n')
                    if len(lines) <= 1:
                        print("No interactions found in STRING")
                        return None, None
                    
                    header = lines[0].split('\t')
                    for line in lines[1:]:
                        parts = line.split('\t')
                        try:
                            edges.append({
                                'gene1': parts[header.index('preferredName_A')],
                                'gene2': parts[header.index('preferredName_B')],
                                'score': float(parts[header.index('score')])
                            })
                        except (IndexError, ValueError, KeyError):
                            continue
                    break
                else:
                    print(f"STRING network query failed (attempt {attempt+1}/{max_retries}): {response.status_code}")
            except Exception as e:
                print(f"STRING network error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
        
        if not edges:
            print("No valid edges retrieved from STRING")
            return None, None
        
        edges_df = pd.DataFrame(edges)
        print(f"Retrieved {len(edges_df)} interactions from STRING")
        
        return edges_df, mapping
    
    def build_ppi_network(self, deg_genes):
        """
        Build a PPI network graph - improved error handling
        """
        if len(deg_genes) == 0:
            print("No DEGs to build network")
            return None
        
        result = self.query_string_network(deg_genes)
        
        if result is None or result[0] is None:
            print("Failed to retrieve STRING network")
            return None
        
        edges_df, mapping = result
        
        # Build NetworkX graph
        G = nx.Graph()
        for _, row in edges_df.iterrows():
            G.add_edge(row['gene1'], row['gene2'], weight=row['score'])
        
        if G.number_of_nodes() == 0:
            print("Network has no nodes")
            return None
        
        return G
    
    def identify_hub_genes(self, G, top_k=20):
        """
        Identify hub genes (core regulatory genes) in the network
        """
        if G is None or G.number_of_nodes() == 0:
            return []
        
        # Compute multiple centrality metrics
        try:
            degree_cent = nx.degree_centrality(G)
            betweenness_cent = nx.betweenness_centrality(G)
            closeness_cent = nx.closeness_centrality(G)
        except Exception as e:
            print(f"Error computing centrality: {e}")
            # Fall back to degree centrality at minimum
            degree_cent = nx.degree_centrality(G)
            betweenness_cent = {node: 0 for node in G.nodes()}
            closeness_cent = {node: 0 for node in G.nodes()}

        # Composite scoring
        hub_scores = {}
        for gene in G.nodes():
            hub_scores[gene] = (
                degree_cent.get(gene, 0) * 0.4 +
                betweenness_cent.get(gene, 0) * 0.3 +
                closeness_cent.get(gene, 0) * 0.3
            )

        # Sort by score
        sorted_hubs = sorted(hub_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_hubs[:top_k]
    
    def perform_pathway_enrichment(self, deg_genes, background_genes=None, max_retries=3):
        """
        Pathway enrichment analysis (using STRING enrichment API) - improved version
        """
        if len(deg_genes) == 0:
            print("No genes for enrichment analysis")
            return None
        
        print(f"Performing pathway enrichment analysis for {len(deg_genes)} genes...")
        
        url = f"{self.string_base}/json/enrichment"
        params = {
            "identifiers": "\r".join(deg_genes[:500]),  # Limit the number of genes
            "species": self.species
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, data=params, timeout=60)
                if response.ok:
                    enrichment_data = response.json()
                    
                    enrichment_results = []
                    for item in enrichment_data:
                        if item.get('category') in ['KEGG', 'Reactome', 'GO Process']:
                            enrichment_results.append({
                                'category': item.get('category', ''),
                                'term': item.get('term', ''),
                                'description': item.get('description', ''),
                                'fdr': float(item.get('fdr', 1.0)),
                                'n_genes': int(item.get('number_of_genes', 0)),
                                'genes': item.get('inputGenes', '')
                            })
                    
                    if enrichment_results:
                        df = pd.DataFrame(enrichment_results)
                        print(f"Found {len(df)} enriched terms")
                        return df
                    else:
                        print("No significant enrichment found")
                        return None
                else:
                    print(f"Enrichment query failed (attempt {attempt+1}/{max_retries}): {response.status_code}")
            except Exception as e:
                print(f"Enrichment error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
        
        return None
    
    def analyze_drug_perturbation(self, adata, drug_id, cell_type=None):
        """
        Full analysis of the perturbation effect of a single drug - improved version
        """
        # Added: convert SMILES to drug name
        drug_name = self.smiles_converter.smiles_to_name(drug_id)
        print(f"\n=== Analyzing drug: {drug_name} ===")
        print(f"    SMILES: {drug_id[:50]}{'...' if len(drug_id) > 50 else ''}")
        
        # Identify perturbation identifier column
        ptrb_col = 'sub_ptrb' if 'sub_ptrb' in adata.obs.columns else 'canonical_smiles'

        # Filter data
        if cell_type:
            mask = (adata.obs[ptrb_col] == drug_id) & (adata.obs['cell_type'] == cell_type)
        else:
            mask = adata.obs[ptrb_col] == drug_id
        
        ctrl_data = adata[mask & (adata.obs['condition'] == 'control')]
        ptrb_data = adata[mask & (adata.obs['condition'] == 'perturb')]
        
        if len(ctrl_data) == 0 or len(ptrb_data) == 0:
            print(f"Insufficient data: {len(ctrl_data)} control, {len(ptrb_data)} perturbed samples")
            return None
        
        print(f"Using {len(ctrl_data)} control and {len(ptrb_data)} perturbed samples")
        
        # Get expression matrices
        ctrl_exp = ctrl_data.X.A if hasattr(ctrl_data.X, 'A') else ctrl_data.X
        ptrb_exp = ptrb_data.X.A if hasattr(ptrb_data.X, 'A') else ptrb_data.X
        gene_names = list(adata.var_names)

        # 1. Identify DEGs
        deg_genes, log2fc, padj = self.get_deg_genes(
            ctrl_exp, ptrb_exp, gene_names,
            log2fc_threshold=0.5,
            padj_threshold=0.1,
            min_genes=20  # Ensure at least 20 genes are returned
        )
        
        if len(deg_genes) == 0:
            print("No DEGs identified - analysis cannot continue")
            return None
        
        # 2. Build PPI network
        G = self.build_ppi_network(deg_genes)
        if G is None:
            print("Failed to build PPI network, but continuing with available data...")
            G = nx.Graph()  # Create empty graph
        else:
            print(f"PPI network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # 3. Identify hub genes
        hub_genes = []
        if G.number_of_nodes() > 0:
            hub_genes = self.identify_hub_genes(G)
            if len(hub_genes) > 0:
                print("\nTop 10 Hub Genes:")
                for gene, score in hub_genes[:10]:
                    print(f"  {gene}: {score:.4f}")

        # 4. Pathway enrichment
        enrichment_df = None
        if len(deg_genes) >= 10:  # Require at least 10 genes for enrichment analysis
            enrichment_df = self.perform_pathway_enrichment(deg_genes)
            if enrichment_df is not None and len(enrichment_df) > 0:
                print(f"\nEnriched pathways: {len(enrichment_df)}")
                print("\nTop 5 enriched pathways:")
                top_pathways = enrichment_df.nsmallest(5, 'fdr')[['category', 'description', 'fdr']]
                print(top_pathways.to_string(index=False))
        
        return {
            'drug_id': drug_id,  # Added: retain original SMILES
            'drug_name': drug_name,  # Added: converted drug name
            'deg_genes': deg_genes,
            'log2fc': log2fc,
            'padj': padj,
            'network': G,
            'hub_genes': hub_genes,
            'enrichment': enrichment_df,
            'n_control': len(ctrl_data),
            'n_perturbed': len(ptrb_data)
        }
    
    def save_hub_genes_to_csv(self, result, save_path):
        """
        Save hub gene analysis results to a CSV file

        Args:
            result: Return value from analyze_drug_perturbation
            save_path: Output file path
        """
        if result is None or not result['hub_genes']:
            print(f"No hub genes to save")
            return None
        
        hub_data = []
        for rank, (gene, score) in enumerate(result['hub_genes'], 1):
            # Retrieve the log2fc and padj for this gene
            gene_idx = None
            if gene in result['deg_genes']:
                gene_idx = result['deg_genes'].index(gene)
            
            hub_data.append({
                'rank': rank,
                'gene': gene,
                'hub_score': score,
                'log2fc': result['log2fc'][gene_idx] if gene_idx is not None else 'N/A',
                'padj': result['padj'][gene_idx] if gene_idx is not None else 'N/A',
                'drug_name': result['drug_name'],
                'drug_smiles': result['drug_id']
            })
        
        hub_df = pd.DataFrame(hub_data)
        hub_df.to_csv(save_path, index=False)
        print(f"Hub genes saved to {save_path}")
        return hub_df
    
    def save_enrichment_to_csv(self, result, save_path):
        """
        Save pathway enrichment analysis results to a CSV file

        Args:
            result: Return value from analyze_drug_perturbation
            save_path: Output file path
        """
        if result is None or result['enrichment'] is None:
            print(f"No enrichment results to save")
            return None
        
        enrichment_df = result['enrichment'].copy()
        enrichment_df['drug_name'] = result['drug_name']
        enrichment_df['drug_smiles'] = result['drug_id']
        
        # Sort by FDR
        enrichment_df = enrichment_df.sort_values('fdr')
        
        enrichment_df.to_csv(save_path, index=False)
        print(f"Enrichment results saved to {save_path}")
        return enrichment_df
    
    def visualize_network(self, G, hub_genes, deg_genes=None, log2fc=None,
                         save_path='drug_network.pdf', layout='spring', drug_name=None):
        """
        Visualize PPI network - improved version (enlarged content and font sizes)

        Args:
            layout: Layout algorithm, options: 'spring', 'kamada_kawai', 'shell', 'spring_large_k'
            drug_name: Drug name (used as plot title)
        """
        if G is None or G.number_of_nodes() == 0:
            print("Cannot visualize empty network")
            return
        
        # Limit network size for visualization
        if G.number_of_nodes() > 100:
            print(f"Network too large ({G.number_of_nodes()} nodes), showing only largest connected component")
            # Get the largest connected component
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            
            # If still too large, keep only the highest-degree nodes
            if G.number_of_nodes() > 100:
                degrees = dict(G.degree())
                top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:100]
                G = G.subgraph([n for n, _ in top_nodes]).copy()
        
        # Enlarge figure size
        plt.figure(figsize=(22, 20))

        # Set node sizes (hub genes are larger) - doubled in size
        hub_gene_names = [g for g, _ in hub_genes] if hub_genes else []
        node_sizes = [6000 if node in hub_gene_names[:10] else 2400 for node in G.nodes()]

        # Node colors - based on log2fc
        if deg_genes and log2fc is not None:
            gene_to_fc = dict(zip(deg_genes, log2fc))
            node_colors = [gene_to_fc.get(node, 0) for node in G.nodes()]
            cmap = plt.cm.RdBu_r
        else:
            node_colors = ['red' if node in hub_gene_names[:10] else 'lightblue' for node in G.nodes()]
            cmap = None
        
        # Choose layout algorithm based on the layout parameter
        try:
            if layout == 'spring':
                # Standard spring layout
                pos = nx.spring_layout(G, k=0.6/np.sqrt(G.number_of_nodes()), iterations=80, seed=42, scale=2.5)
            elif layout == 'spring_large_k':
                # Spring layout with larger k value for wider node spacing
                pos = nx.spring_layout(G, k=3.0/np.sqrt(G.number_of_nodes()), iterations=200, seed=42, scale=3.0)
            elif layout == 'kamada_kawai':
                # Kamada-Kawai layout based on graph-theoretic distances, more even distribution
                pos = nx.kamada_kawai_layout(G, scale=2.5)
            elif layout == 'shell':
                # Shell layout: hub nodes at the center, other nodes arranged in concentric shells
                # Shell 1: top 10 hub genes
                # Shell 2: remaining hub genes (rank 11-20)
                # Shell 3: all other nodes
                shell1 = [n for n in hub_gene_names[:10] if n in G.nodes()]
                shell2 = [n for n in hub_gene_names[10:20] if n in G.nodes()]
                shell3 = [n for n in G.nodes() if n not in shell1 and n not in shell2]
                shells = [shell1, shell2, shell3] if shell1 else [list(G.nodes())]
                # Filter out empty shells
                shells = [s for s in shells if len(s) > 0]
                pos = nx.shell_layout(G, nlist=shells, scale=2.5)
            else:
                pos = nx.spring_layout(G, k=0.6/np.sqrt(G.number_of_nodes()), iterations=80, seed=42, scale=2.5)
        except Exception as e:
            print(f"Layout {layout} failed: {e}, falling back to random layout")
            pos = nx.random_layout(G, seed=42)
        
        # Draw edges - with thicker lines
        nx.draw_networkx_edges(G, pos, alpha=0.4, width=2.5, edge_color='gray')

        # Draw nodes (without colorbar)
        if cmap:
            nodes = nx.draw_networkx_nodes(
                G, pos, node_size=node_sizes, node_color=node_colors,
                cmap=cmap, alpha=0.9, vmin=-2, vmax=2, edgecolors='black', linewidths=2.0
            )
        else:
            nx.draw_networkx_nodes(
                G, pos, node_size=node_sizes, node_color=node_colors, 
                alpha=0.85, edgecolors='black', linewidths=2.0
            )
        
        # Label hub genes - use labels parameter to avoid KeyError
        if hub_gene_names:
            labels_to_draw = {n: n for n in hub_gene_names[:20] if n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=56, font_weight='bold', font_color='black')
        
        # Modified: use drug name as plot title
        layout_names = {
            'spring': 'Spring Layout',
            'spring_large_k': 'Spring Layout (large k)',
            'kamada_kawai': 'Kamada-Kawai Layout',
            'shell': 'Shell Layout'
        }
        layout_name = layout_names.get(layout, layout)
        
        # Added: use drug name as title if provided
        if drug_name:
            title = f'{drug_name}\nPPI Network - {layout_name}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)'
        else:
            title = f'Drug Perturbation PPI Network - {layout_name}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)\nRed nodes: Top Hub genes'
        
        plt.title(title, fontsize=48, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        try:
            plt.savefig(save_path, format='pdf', bbox_inches='tight', facecolor='white')
            print(f"Network visualization saved to {save_path}")
        except Exception as e:
            print(f"Failed to save visualization: {e}")
        
        plt.close()
    
    def visualize_network_all_layouts(self, G, hub_genes, deg_genes=None, log2fc=None,
                                      save_dir='layouts', drug_name=None):
        """
        Generate network plots using all layout algorithms for easy comparison

        Args:
            G: NetworkX graph
            hub_genes: List of hub genes
            deg_genes: List of DEG genes
            log2fc: Log2 fold change values
            save_dir: Output directory
            drug_name: Drug name (used as plot title)
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        layouts = ['kamada_kawai', 'shell']
        
        for layout in layouts:
            print(f"\nGenerating {layout} layout...")
            save_path = f"{save_dir}/network_{layout}.pdf"
            self.visualize_network(G, hub_genes, deg_genes, log2fc, save_path, layout=layout, drug_name=drug_name)
        
        # Also generate a colorbar
        self.save_colorbar(f"{save_dir}/colorbar.pdf")
        
        print(f"\nAll layouts saved to {save_dir}/")
    
    def save_colorbar(self, save_path='colorbar.pdf', vmin=-2, vmax=2):
        """
        Generate a standalone colorbar PDF

        Args:
            save_path: Output file path
            vmin: Minimum value of the colorbar
            vmax: Maximum value of the colorbar
        """
        fig, ax = plt.subplots(figsize=(2, 10))

        # Create colorbar
        cmap = plt.cm.RdBu_r
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(sm, cax=ax)
        cbar.ax.tick_params(labelsize=48)
        cbar.set_label('Log2 Fold Change', fontsize=56, fontweight='bold')
        
        plt.tight_layout()
        
        try:
            plt.savefig(save_path, format='pdf', bbox_inches='tight', facecolor='white')
            print(f"Colorbar saved to {save_path}")
        except Exception as e:
            print(f"Failed to save colorbar: {e}")
        
        plt.close()
    
    def batch_analyze_drugs(self, adata, n_drugs=10, cell_type=None, save_dir='drug_analysis'):
        """
        Batch analysis of multiple drugs
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        ptrb_col = 'sub_ptrb' if 'sub_ptrb' in adata.obs.columns else 'canonical_smiles'
        
        if cell_type:
            drugs = adata.obs[adata.obs['cell_type'] == cell_type][ptrb_col].unique()
        else:
            drugs = adata.obs[ptrb_col].unique()
        
        drugs = drugs[:n_drugs]
        
        # Added: save SMILES conversion table
        conversion_df = self.smiles_converter.save_conversion_table(
            list(drugs), 
            save_path=f"{save_dir}/smiles_conversion.csv"
        )
        
        results_summary = []
        all_hub_genes = []  # Added: aggregate hub genes across all drugs
        all_enrichments = []  # Added: aggregate enrichment results across all drugs

        # Generate a standalone colorbar PDF
        colorbar_path = f"{save_dir}/colorbar.pdf"
        self.save_colorbar(colorbar_path)
        
        for i, drug in enumerate(drugs):
            print(f"\n{'='*60}")
            print(f"Processing drug {i+1}/{len(drugs)}")
            print(f"{'='*60}")
            
            result = self.analyze_drug_perturbation(adata, drug, cell_type)
            
            if result is not None:
                # Save network plots - generate all layout variants using drug name
                if result['network'] is not None and result['network'].number_of_nodes() > 0:
                    drug_layout_dir = f"{save_dir}/drug_{i+1}_layouts"
                    self.visualize_network_all_layouts(
                        result['network'],
                        result['hub_genes'],
                        result['deg_genes'],
                        result['log2fc'],
                        save_dir=drug_layout_dir,
                        drug_name=result['drug_name']  # Added: pass drug name
                    )
                
                # Added: save hub genes for individual drug to CSV
                hub_csv_path = f"{save_dir}/drug_{i+1}_hub_genes.csv"
                hub_df = self.save_hub_genes_to_csv(result, hub_csv_path)
                if hub_df is not None:
                    all_hub_genes.append(hub_df)

                # Added: save enrichment results for individual drug to CSV
                enrichment_csv_path = f"{save_dir}/drug_{i+1}_enrichment.csv"
                enrich_df = self.save_enrichment_to_csv(result, enrichment_csv_path)
                if enrich_df is not None:
                    all_enrichments.append(enrich_df)

                # Summary entry - Added: drug name
                results_summary.append({
                    'drug_id': drug[:50],
                    'drug_name': result['drug_name'],  # Added: drug name
                    'n_degs': len(result['deg_genes']),
                    'n_network_nodes': result['network'].number_of_nodes() if result['network'] else 0,
                    'n_network_edges': result['network'].number_of_edges() if result['network'] else 0,
                    'top_hub': result['hub_genes'][0][0] if result['hub_genes'] else 'N/A',
                    'n_enriched_pathways': len(result['enrichment']) if result['enrichment'] is not None else 0,
                    'n_control_samples': result['n_control'],
                    'n_perturbed_samples': result['n_perturbed']
                })
        
        # Save summary results
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv(f"{save_dir}/analysis_summary.csv", index=False)
        print(f"\nAnalysis summary saved to {save_dir}/analysis_summary.csv")

        # Added: save aggregated hub genes across all drugs
        if all_hub_genes:
            all_hub_df = pd.concat(all_hub_genes, ignore_index=True)
            all_hub_df.to_csv(f"{save_dir}/all_hub_genes.csv", index=False)
            print(f"All hub genes saved to {save_dir}/all_hub_genes.csv")

        # Added: save aggregated enrichment results across all drugs
        if all_enrichments:
            all_enrich_df = pd.concat(all_enrichments, ignore_index=True)
            all_enrich_df.to_csv(f"{save_dir}/all_enrichments.csv", index=False)
            print(f"All enrichment results saved to {save_dir}/all_enrichments.csv")
        
        return summary_df