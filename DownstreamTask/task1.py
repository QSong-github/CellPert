import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from torch_geometric.data import DataLoader
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')
seed = 8
np.random.seed(seed)


class CrossDatasetConsistencyAnalysis:
    """
    Task 1: Cross-Dataset Consistency Validation (LINCS vs Tahoe)

    Challenges:
    - LINCS uses canonical_smiles, Tahoe uses sub_ptrb (both are SMILES format)
    - Cell types differ (LINCS uses standard names, Tahoe uses Cellosaurus IDs)

    Strategies:
    1. SMILES exact matching: find drugs with identical chemical structures
    2. Morgan fingerprint similarity: find chemically similar drugs even when SMILES differ
    3. Latent space alignment: validate representation consistency across datasets
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
        # Prediction cache: (dataset_key, idx) -> {'pred': array, 'latent': array}
        self._cache = {}

    def _precompute_all(self, lincs_test, lincs_adata, tahoe_tests, tahoe_adatas, batch_size=32):
        """
        Precompute and cache all predictions and latent embeddings for every sample
        using batched inference.  All three strategies then do dict lookups instead
        of re-running the model.
        """
        print("\n" + "="*70)
        print("Precomputing model predictions for all samples (batched)...")
        print("="*70)

        def process_dataset(test_dataset, n_samples, dataset_key, desc):
            uncached = [i for i in range(n_samples)
                        if (dataset_key, i) not in self._cache]
            if not uncached:
                print(f"  {desc}: all {n_samples} samples already cached — skipping")
                return

            print(f"  {desc}: {len(uncached)} samples to process")
            for batch_start in tqdm(range(0, len(uncached), batch_size), desc=desc):
                batch_indices = uncached[batch_start:batch_start + batch_size]
                samples, valid_idx = [], []
                for idx in batch_indices:
                    try:
                        samples.append(test_dataset[int(idx)])
                        valid_idx.append(idx)
                    except Exception:
                        continue

                if not samples:
                    continue

                try:
                    loader = DataLoader(samples, batch_size=len(samples), shuffle=False)
                    data = next(iter(loader)).to(self.device)
                    n = len(valid_idx)

                    with torch.no_grad():
                        preds, _ = self.model.predict(data)
                        _, _, graph_mu, _ = self.model.encoder(
                            data.x, data.edge_index, data.batch
                        )

                    # graph_mu is (n, latent_dim) after global pooling — one row per graph
                    # preds is (n, n_genes) if per-graph, or (n_nodes, n_genes) if per-node
                    if preds.shape[0] == n:
                        preds_list = [preds[i].cpu().numpy().flatten() for i in range(n)]
                    else:
                        preds_list = [
                            preds[data.batch == i].cpu().numpy().flatten()
                            for i in range(n)
                        ]

                    for j, idx in enumerate(valid_idx):
                        self._cache[(dataset_key, idx)] = {
                            'pred':   preds_list[j],
                            'latent': graph_mu[j].cpu().numpy().flatten(),
                        }

                except Exception:
                    # Fallback: individual inference so a single bad sample can't
                    # block the whole batch
                    for idx in batch_indices:
                        if (dataset_key, idx) in self._cache:
                            continue
                        try:
                            sample = test_dataset[int(idx)]
                            loader = DataLoader([sample], batch_size=1, shuffle=False)
                            data = next(iter(loader)).to(self.device)
                            with torch.no_grad():
                                pred, _ = self.model.predict(data)
                                _, _, graph_mu, _ = self.model.encoder(
                                    data.x, data.edge_index, data.batch
                                )
                            self._cache[(dataset_key, idx)] = {
                                'pred':   pred.cpu().numpy().flatten(),
                                'latent': graph_mu.cpu().numpy().flatten(),
                            }
                        except Exception:
                            continue

        process_dataset(lincs_test, len(lincs_adata), 'lincs', 'LINCS')
        for plate_idx, (tahoe_test, tahoe_adata) in enumerate(zip(tahoe_tests, tahoe_adatas)):
            process_dataset(
                tahoe_test, len(tahoe_adata),
                f'tahoe_{plate_idx}', f'Tahoe P{plate_idx+1}'
            )

        print(f"\n✓ Cache ready: {len(self._cache)} entries total")
    
    def strategy1_smiles_exact_matching(self, lincs_adata, tahoe_adatas, 
                                        lincs_test, tahoe_tests, save_dir):
        """
        Strategy 1: SMILES exact matching
        LINCS canonical_smiles vs Tahoe sub_ptrb
        Saves correlation results independently for each sample pair
        """
        print("\n" + "="*70)
        print("Strategy 1: Cross-Dataset SMILES Exact Matching")
        print("="*70)
        
        # LINCS SMILES
        lincs_ptrb_mask = lincs_adata.obs['condition'] == 'perturb'
        lincs_smiles = set(lincs_adata.obs[lincs_ptrb_mask]['canonical_smiles'].unique())
        print(f"LINCS: {len(lincs_smiles)} unique drugs (canonical_smiles)")
        
        # Tahoe SMILES (merged across all plates)
        tahoe_smiles = set()
        tahoe_smiles_by_plate = {}
        
        for i, adata in enumerate(tahoe_adatas):
            ptrb_mask = adata.obs['condition'] == 'perturb'
            plate_smiles = set(adata.obs[ptrb_mask]['sub_ptrb'].unique())
            tahoe_smiles.update(plate_smiles)
            tahoe_smiles_by_plate[i] = plate_smiles
            print(f"Tahoe P{i+1}: {len(plate_smiles)} unique drugs (sub_ptrb)")
        
        print(f"Tahoe (all plates): {len(tahoe_smiles)} unique drugs")
        
        # Find common SMILES
        common_smiles = lincs_smiles & tahoe_smiles
        print(f"\n✓ Common drugs (exact SMILES match): {len(common_smiles)}")
        
        if len(common_smiles) > 0:
            print(f"  Example matches:")
            for smiles in list(common_smiles)[:3]:
                print(f"    {smiles[:60]}...")
        
        if len(common_smiles) == 0:
            print("\n⚠ No exact SMILES matches found between LINCS and Tahoe")
            print("   This suggests they use different chemical libraries.")
            print("   Will use alternative strategies for validation...")
            return pd.DataFrame(), common_smiles
        
        # Save results for each sample pair
        results = []
        common_smiles_list = list(common_smiles)[:30]  # at most 30
        
        print(f"\nAnalyzing {len(common_smiles_list)} matched drugs...")
        
        for smiles in tqdm(common_smiles_list, desc="Processing matched drugs"):
            # LINCS prediction
            lincs_mask = (lincs_adata.obs['canonical_smiles'] == smiles) & \
                        (lincs_adata.obs['condition'] == 'control')
            
            if lincs_mask.sum() == 0:
                continue
            
            lincs_indices = np.where(lincs_mask)[0][:5]
            lincs_preds = [
                self._cache[('lincs', int(idx))]['pred']
                for idx in lincs_indices
                if ('lincs', int(idx)) in self._cache
            ]

            if len(lincs_preds) == 0:
                continue

            # Tahoe predictions (check all plates)
            for plate_idx, tahoe_adata in enumerate(tahoe_adatas):
                if smiles not in tahoe_smiles_by_plate[plate_idx]:
                    continue

                tahoe_mask = (tahoe_adata.obs['sub_ptrb'] == smiles) & \
                            (tahoe_adata.obs['condition'] == 'control')

                if tahoe_mask.sum() == 0:
                    continue

                tahoe_indices = np.where(tahoe_mask)[0][:5]
                tahoe_preds = [
                    self._cache[(f'tahoe_{plate_idx}', int(idx))]['pred']
                    for idx in tahoe_indices
                    if (f'tahoe_{plate_idx}', int(idx)) in self._cache
                ]
                
                # Compute correlation between each LINCS sample and each Tahoe sample
                if len(tahoe_preds) > 0:
                    for i, lincs_pred in enumerate(lincs_preds):
                        for j, tahoe_pred in enumerate(tahoe_preds):
                            try:
                                pearson_r, _ = pearsonr(lincs_pred, tahoe_pred)
                                spearman_r, _ = spearmanr(lincs_pred, tahoe_pred)
                                
                                results.append({
                                    'drug_smiles': smiles,
                                    'dataset1': 'LINCS',
                                    'dataset2': f'Tahoe_P{plate_idx+1}',
                                    'lincs_sample_idx': i,
                                    'tahoe_sample_idx': j,
                                    'pearson_correlation': pearson_r,
                                    'spearman_correlation': spearman_r
                                })
                            except:
                                continue
        
        if len(results) == 0:
            print("No successful cross-dataset comparisons")
            return pd.DataFrame(), common_smiles
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{save_dir}/cross_dataset_smiles_exact_all_samples.csv', index=False)
        
        print(f"\n✓ Saved {len(results)} sample pair correlations")
        
        # Compute average results for summary
        avg_results = results_df.groupby(['drug_smiles', 'dataset1', 'dataset2']).agg({
            'pearson_correlation': 'mean',
            'spearman_correlation': 'mean'
        }).reset_index()
        avg_results.to_csv(f'{save_dir}/cross_dataset_smiles_exact_avg.csv', index=False)
        
        return results_df, common_smiles

    def strategy2_morgan_fingerprint_similarity(self, lincs_adata, tahoe_adatas,
                                               lincs_test, tahoe_tests, save_dir,
                                               similarity_threshold=0.85):
        """
        Strategy 2: Morgan fingerprint similarity matching
        When SMILES are not identical, use molecular fingerprints to find similar drugs
        """
        print("\n" + "="*70)
        print("Strategy 2: Morgan Fingerprint Similarity Matching")
        print("="*70)
        
        if 'morgan_fp' not in lincs_adata.obs.columns or \
           'morgan_fp' not in tahoe_adatas[0].obs.columns:
            print("⚠ Morgan fingerprints not available, skipping this strategy")
            return pd.DataFrame()
        
        print(f"Using similarity threshold: {similarity_threshold}")
        
        # Sample some LINCS drugs
        lincs_ptrb_mask = lincs_adata.obs['condition'] == 'perturb'
        lincs_drugs = lincs_adata.obs[lincs_ptrb_mask][['canonical_smiles', 'morgan_fp']].drop_duplicates()
        lincs_drugs = lincs_drugs.sample(min(50, len(lincs_drugs)))  # at most 50
        
        print(f"Sampled {len(lincs_drugs)} LINCS drugs")
        
        # Sample some Tahoe drugs (from all plates)
        tahoe_drugs_list = []
        for i, adata in enumerate(tahoe_adatas):
            ptrb_mask = adata.obs['condition'] == 'perturb'
            plate_drugs = adata.obs[ptrb_mask][['sub_ptrb', 'morgan_fp']].drop_duplicates()
            plate_drugs['plate_idx'] = i
            tahoe_drugs_list.append(plate_drugs)
        
        tahoe_drugs = pd.concat(tahoe_drugs_list, ignore_index=True)
        print(f"Collected {len(tahoe_drugs)} unique Tahoe drugs")
        
        # Compute similarity and find matches
        def tanimoto_similarity(fp1, fp2):
            """Compute Tanimoto similarity"""
            fp1_bits = set([i for i, bit in enumerate(fp1) if bit == '1'])
            fp2_bits = set([i for i, bit in enumerate(fp2) if bit == '1'])
            
            intersection = len(fp1_bits & fp2_bits)
            union = len(fp1_bits | fp2_bits)
            
            return intersection / union if union > 0 else 0.0
        
        matched_pairs = []
        
        print("\nFinding similar drug pairs...")
        for _, lincs_row in tqdm(lincs_drugs.iterrows(), total=len(lincs_drugs), 
                                desc="Matching drugs"):
            lincs_fp = lincs_row['morgan_fp']
            lincs_smiles = lincs_row['canonical_smiles']
            
            for _, tahoe_row in tahoe_drugs.iterrows():
                tahoe_fp = tahoe_row['morgan_fp']
                tahoe_smiles = tahoe_row['sub_ptrb']
                
                # Skip identical SMILES (already handled in Strategy 1)
                if lincs_smiles == tahoe_smiles:
                    continue
                
                similarity = tanimoto_similarity(lincs_fp, tahoe_fp)
                
                if similarity >= similarity_threshold:
                    matched_pairs.append({
                        'lincs_smiles': lincs_smiles,
                        'tahoe_smiles': tahoe_smiles,
                        'plate_idx': tahoe_row['plate_idx'],
                        'similarity': similarity
                    })
        
        print(f"\n✓ Found {len(matched_pairs)} similar drug pairs (similarity ≥ {similarity_threshold})")
        
        if len(matched_pairs) == 0:
            print("   No similar drugs found with current threshold")
            return pd.DataFrame()
        
        # Analyze prediction consistency for these similar drug pairs
        results = []

        for pair in tqdm(matched_pairs[:20], desc="Analyzing similar pairs"):  # at most 20 pairs
            lincs_smiles = pair['lincs_smiles']
            tahoe_smiles = pair['tahoe_smiles']
            plate_idx = pair['plate_idx']

            # LINCS prediction
            lincs_mask = (lincs_adata.obs['canonical_smiles'] == lincs_smiles) & \
                        (lincs_adata.obs['condition'] == 'control')
            
            if lincs_mask.sum() == 0:
                continue
            
            lincs_indices = np.where(lincs_mask)[0][:3]
            lincs_preds = [
                self._cache[('lincs', int(idx))]['pred']
                for idx in lincs_indices
                if ('lincs', int(idx)) in self._cache
            ]

            # Tahoe prediction
            tahoe_adata = tahoe_adatas[plate_idx]
            tahoe_mask = (tahoe_adata.obs['sub_ptrb'] == tahoe_smiles) & \
                        (tahoe_adata.obs['condition'] == 'control')

            if tahoe_mask.sum() == 0:
                continue

            tahoe_indices = np.where(tahoe_mask)[0][:3]
            tahoe_preds = [
                self._cache[(f'tahoe_{plate_idx}', int(idx))]['pred']
                for idx in tahoe_indices
                if (f'tahoe_{plate_idx}', int(idx)) in self._cache
            ]
            
            # Compute correlation
            if len(lincs_preds) > 0 and len(tahoe_preds) > 0:
                lincs_avg = np.mean(lincs_preds, axis=0)
                tahoe_avg = np.mean(tahoe_preds, axis=0)
                
                try:
                    pearson_r, _ = pearsonr(lincs_avg, tahoe_avg)
                    spearman_r, _ = spearmanr(lincs_avg, tahoe_avg)
                    
                    results.append({
                        'lincs_drug': lincs_smiles[:40],
                        'tahoe_drug': tahoe_smiles[:40],
                        'chemical_similarity': pair['similarity'],
                        'dataset1': 'LINCS',
                        'dataset2': f'Tahoe_P{plate_idx+1}',
                        'pearson_correlation': pearson_r,
                        'spearman_correlation': spearman_r
                    })
                except:
                    continue
        
        if len(results) == 0:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{save_dir}/cross_dataset_morgan_similarity.csv', index=False)
        
        return results_df

    def strategy3_latent_space_alignment(self, lincs_adata, tahoe_adatas,
                                            lincs_test, tahoe_tests, save_dir):
            """
            Strategy 3: Latent space alignment analysis (improved version)

            Core idea:
            1. Compare the distribution of raw expression vs latent embedding
            2. If the model learns effectively, the latent space should align the two datasets
               better than the raw space
            3. Use arrow plots to show KS and Silhouette metric improvements per plate

            Note: This version uses all data without subsampling
            """
            print("\n" + "="*70)
            print("Strategy 3: Cross-Dataset Latent Space Alignment (Full Data)")
            print("="*70)
            
            # ============================================================
            # Part 1: Collect all LINCS data (as reference)
            # ============================================================
            print("\n[Part 1] Collecting ALL LINCS samples as reference...")
            
            lincs_raw = []
            lincs_latents = []

            n_lincs = len(lincs_adata)  # use all data

            for idx in range(n_lincs):
                key = ('lincs', idx)
                if key not in self._cache:
                    continue
                lincs_latents.append(self._cache[key]['latent'])
                raw_expr = lincs_adata.X[idx].toarray().flatten() \
                    if hasattr(lincs_adata.X, 'toarray') else lincs_adata.X[idx].flatten()
                lincs_raw.append(raw_expr)

            lincs_raw = np.array(lincs_raw)
            lincs_latents = np.array(lincs_latents)

            print(f"  Collected {len(lincs_raw)} / {n_lincs} LINCS samples (from cache)")
            
            # ============================================================
            # Part 2: Compute metrics for each Tahoe plate (using all data)
            # ============================================================
            print("\n[Part 2] Computing metrics for each Tahoe plate (full data)...")
            
            plate_metrics = []
            
            for plate_idx, tahoe_adata in enumerate(tahoe_adatas):
                plate_name = f'P{plate_idx+1}'
                print(f"  Processing {plate_name} ({len(tahoe_adata)} samples)...")
                
                # Collect all data for this plate from cache
                tahoe_raw = []
                tahoe_latents = []

                n_samples = len(tahoe_adata)  # use all data
                dataset_key = f'tahoe_{plate_idx}'

                for idx in range(n_samples):
                    key = (dataset_key, idx)
                    if key not in self._cache:
                        continue
                    tahoe_latents.append(self._cache[key]['latent'])
                    raw_expr = tahoe_adata.X[idx].toarray().flatten() \
                        if hasattr(tahoe_adata.X, 'toarray') else tahoe_adata.X[idx].flatten()
                    tahoe_raw.append(raw_expr)
                
                if len(tahoe_raw) < 10:
                    print(f"    Skipping {plate_name}: insufficient samples ({len(tahoe_raw)})")
                    continue
                
                tahoe_raw = np.array(tahoe_raw)
                tahoe_latents = np.array(tahoe_latents)
                
                print(f"    Collected {len(tahoe_raw)} / {n_samples} samples")
                
                # Compute KS statistic (using all dimensions)
                def compute_ks_statistic(data1, data2):
                    """Compute the mean KS statistic across all dimensions"""
                    n_dims = min(data1.shape[1], data2.shape[1])
                    ks_stats = []
                    for d in range(n_dims):  # iterate over all dimensions
                        stat, _ = ks_2samp(data1[:, d], data2[:, d])
                        ks_stats.append(stat)
                    return np.mean(ks_stats)
                
                # Compute Silhouette score (using all data)
                def compute_silhouette(data1, data2):
                    """Compute Silhouette score using all data"""
                    combined = np.vstack([data1, data2])
                    labels = np.array([0]*len(data1) + [1]*len(data2))
                    
                    try:
                        score = silhouette_score(combined, labels, metric='cosine')
                    except:
                        score = np.nan
                    return score
                
                # Raw space metrics
                ks_raw = compute_ks_statistic(lincs_raw, tahoe_raw)
                sil_raw = compute_silhouette(lincs_raw, tahoe_raw)
                
                # Latent space metrics
                ks_latent = compute_ks_statistic(lincs_latents, tahoe_latents)
                sil_latent = compute_silhouette(lincs_latents, tahoe_latents)
                
                # Centroid similarity (cosine similarity between dataset centroids)
                def compute_centroid_similarity(data1, data2):
                    """Compute cosine similarity between the centroids of two datasets"""
                    centroid1 = np.mean(data1, axis=0)
                    centroid2 = np.mean(data2, axis=0)
                    
                    # Cosine similarity
                    dot_product = np.dot(centroid1, centroid2)
                    norm1 = np.linalg.norm(centroid1)
                    norm2 = np.linalg.norm(centroid2)
                    
                    if norm1 == 0 or norm2 == 0:
                        return 0.0
                    return dot_product / (norm1 * norm2)
                
                centroid_sim_raw = compute_centroid_similarity(lincs_raw, tahoe_raw)
                centroid_sim_latent = compute_centroid_similarity(lincs_latents, tahoe_latents)
                
                plate_metrics.append({
                    'plate': plate_name,
                    'ks_raw': ks_raw,
                    'ks_latent': ks_latent,
                    'ks_improvement': ks_raw - ks_latent,  # positive = latent is better
                    'silhouette_raw': sil_raw,
                    'silhouette_latent': sil_latent,
                    'silhouette_improvement': sil_raw - sil_latent,  # positive = latent is better (less batch effect)
                    'centroid_sim_raw': centroid_sim_raw,
                    'centroid_sim_latent': centroid_sim_latent,
                    'centroid_sim_improvement': centroid_sim_latent - centroid_sim_raw,  # positive = latent is better (more similar)
                    'n_samples': len(tahoe_raw)
                })
                
                print(f"    KS: {ks_raw:.4f} → {ks_latent:.4f} (Δ={ks_raw-ks_latent:+.4f})")
                print(f"    Silhouette: {sil_raw:.4f} → {sil_latent:.4f} (Δ={sil_raw-sil_latent:+.4f})")
                print(f"    Centroid Sim: {centroid_sim_raw:.4f} → {centroid_sim_latent:.4f} (Δ={centroid_sim_latent-centroid_sim_raw:+.4f})")
            
            if len(plate_metrics) == 0:
                print("⚠ No valid plate data collected")
                return {}
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame(plate_metrics)
            metrics_df.to_csv(f'{save_dir}/latent_alignment_per_plate.csv', index=False)
            
            # ============================================================
            # Part 3: Summary statistics
            # ============================================================
            print("\n[Part 3] Summary Statistics...")
            
            summary = {
                'n_plates': len(plate_metrics),
                'lincs_samples': len(lincs_raw),
                'total_tahoe_samples': int(metrics_df['n_samples'].sum()),
                
                # KS statistics
                'ks_raw_mean': metrics_df['ks_raw'].mean(),
                'ks_latent_mean': metrics_df['ks_latent'].mean(),
                'ks_improvement_mean': metrics_df['ks_improvement'].mean(),
                'ks_improved_count': int((metrics_df['ks_improvement'] > 0).sum()),
                
                # Silhouette statistics
                'silhouette_raw_mean': metrics_df['silhouette_raw'].mean(),
                'silhouette_latent_mean': metrics_df['silhouette_latent'].mean(),
                'silhouette_improvement_mean': metrics_df['silhouette_improvement'].mean(),
                'silhouette_improved_count': int((metrics_df['silhouette_improvement'] > 0).sum()),
                
                # Centroid similarity statistics
                'centroid_sim_raw_mean': metrics_df['centroid_sim_raw'].mean(),
                'centroid_sim_latent_mean': metrics_df['centroid_sim_latent'].mean(),
                'centroid_sim_improvement_mean': metrics_df['centroid_sim_improvement'].mean(),
                'centroid_sim_improved_count': int((metrics_df['centroid_sim_improvement'] > 0).sum()),
            }
            
            # Save summary statistics
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(f'{save_dir}/latent_alignment_summary.csv', index=False)
            
            print("\n" + "-"*60)
            print("Cross-Dataset Alignment Summary (Full Data)")
            print("-"*60)
            print(f"LINCS samples: {summary['lincs_samples']}")
            print(f"Total Tahoe samples: {summary['total_tahoe_samples']}")
            print(f"Plates analyzed: {summary['n_plates']}")
            print(f"\nKS Statistic (↓ better = more similar distributions):")
            print(f"  Raw: {summary['ks_raw_mean']:.4f}, Latent: {summary['ks_latent_mean']:.4f}")
            print(f"  Improved plates: {summary['ks_improved_count']}/{summary['n_plates']}")
            print(f"\nSilhouette (↓ better = less batch effect):")
            print(f"  Raw: {summary['silhouette_raw_mean']:.4f}, Latent: {summary['silhouette_latent_mean']:.4f}")
            print(f"  Improved plates: {summary['silhouette_improved_count']}/{summary['n_plates']}")
            print(f"\nCentroid Similarity (↑ better = more aligned centroids):")
            print(f"  Raw: {summary['centroid_sim_raw_mean']:.4f}, Latent: {summary['centroid_sim_latent_mean']:.4f}")
            print(f"  Improved plates: {summary['centroid_sim_improved_count']}/{summary['n_plates']}")
            print("-"*60)
            
            return summary
        
    def comprehensive_analysis(self, lincs_adata, tahoe_adatas,
                              lincs_test, tahoe_tests,
                              save_dir='./output/task1_cross_dataset'):
        """
        Comprehensive cross-dataset analysis
        """
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "="*70)
        print("COMPREHENSIVE CROSS-DATASET ANALYSIS")
        print("LINCS vs Tahoe")
        print("="*70)

        # Precompute all model predictions once — reused by all three strategies
        self._precompute_all(lincs_test, lincs_adata, tahoe_tests, tahoe_adatas)

        all_metrics = {}
        
        # Strategy 1: SMILES exact matching
        try:
            print("\n" + "▶"*35)
            results_s1, common_smiles = self.strategy1_smiles_exact_matching(
                lincs_adata, tahoe_adatas, lincs_test, tahoe_tests, save_dir
            )
            if len(results_s1) > 0:
                all_metrics['smiles_exact'] = {
                    'mean_pearson': results_s1['pearson_correlation'].mean(),
                    'std_pearson': results_s1['pearson_correlation'].std(),
                    'n_matched_drugs': len(common_smiles),
                    'n_comparisons': len(results_s1)
                }
        except Exception as e:
            print(f"⚠ Strategy 1 failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Strategy 2: Morgan fingerprint similarity
        try:
            print("\n" + "▶"*35)
            results_s2 = self.strategy2_morgan_fingerprint_similarity(
                lincs_adata, tahoe_adatas, lincs_test, tahoe_tests, save_dir
            )
            if len(results_s2) > 0:
                all_metrics['morgan_similarity'] = {
                    'mean_pearson': results_s2['pearson_correlation'].mean(),
                    'n_similar_pairs': len(results_s2)
                }
        except Exception as e:
            print(f"⚠ Strategy 2 failed: {e}")
        
        # Strategy 3: Latent space alignment
        try:
            print("\n" + "▶"*35)
            results_s3 = self.strategy3_latent_space_alignment(
                lincs_adata, tahoe_adatas, lincs_test, tahoe_tests, save_dir
            )
            if results_s3:
                all_metrics['latent_alignment'] = results_s3
        except Exception as e:
            print(f"⚠ Strategy 3 failed: {e}")
        
        # Save summary metrics
        import json
        with open(f'{save_dir}/all_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        return all_metrics