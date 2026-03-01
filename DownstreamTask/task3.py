from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
from torch_geometric.data import DataLoader
import warnings
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
warnings.filterwarnings('ignore')


class DrugRepurposing:
    """
    Task 3: Drug Repurposing
    Predict new indications for known drugs by comparing drug perturbation signatures
    """

    def __init__(self, model, device, output_dir='/blue/qsong1/wang.qing/scPerturb/scUP/scGP/downstreamTask/output/task3_drug_rep'):
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.drug_metadata = None  # Store drug metadata for later use

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def smiles_to_drug_name(self, smiles):
        """
        Attempt to convert SMILES to a drug name.
        Uses RDKit to retrieve molecular information, returning the IUPAC name
        or molecular formula if possible.

        Args:
            smiles: SMILES string

        Returns:
            Drug name or molecular formula; returns truncated SMILES on failure
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles[:30] + "..." if len(smiles) > 30 else smiles

            # Try to retrieve the molecule name if available
            if mol.HasProp('_Name') and mol.GetProp('_Name'):
                return mol.GetProp('_Name')

            # Fall back to molecular formula
            formula = Chem.rdMolDescriptors.CalcMolFormula(mol)

            # Get molecular weight
            mol_weight = Descriptors.MolWt(mol)

            # Return format: MolecularFormula_MolecularWeight
            return f"{formula}_MW{mol_weight:.1f}"

        except Exception as e:
            # Return truncated SMILES if conversion fails
            return smiles[:30] + "..." if len(smiles) > 30 else smiles

    def batch_convert_smiles_to_names(self, smiles_list):
        """
        Batch-convert a list of SMILES strings to drug names.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary mapping SMILES to drug names
        """
        print("Converting SMILES to drug names...")
        smiles_to_name = {}

        for smiles in tqdm(smiles_list, desc="Processing SMILES"):
            smiles_to_name[smiles] = self.smiles_to_drug_name(smiles)

        # Save conversion mapping to CSV
        mapping_df = pd.DataFrame({
            'smiles': list(smiles_to_name.keys()),
            'drug_name': list(smiles_to_name.values())
        })
        mapping_path = os.path.join(self.output_dir, 'smiles_to_name_mapping.csv')
        mapping_df.to_csv(mapping_path, index=False)
        print(f"SMILES to name mapping saved to {mapping_path}")

        return smiles_to_name

    def extract_drug_signatures(self, adata, test_dataset, max_drugs=None, convert_smiles=True):
        """
        Extract the perturbation signature for each drug (based on model predictions).

        Args:
            adata: AnnData object
            test_dataset: Test dataset
            max_drugs: Maximum number of drugs to process
            convert_smiles: Whether to convert SMILES to drug names
        """
        print("Extracting drug perturbation signatures...")

        # Identify the perturbation identifier column
        ptrb_col = 'sub_ptrb' if 'sub_ptrb' in adata.obs.columns else 'canonical_smiles'
        unique_drugs = adata.obs[ptrb_col].unique()

        if max_drugs:
            unique_drugs = unique_drugs[:max_drugs]

        # Convert SMILES to drug names if requested
        smiles_to_name = {}
        if convert_smiles and ptrb_col == 'canonical_smiles':
            smiles_to_name = self.batch_convert_smiles_to_names(unique_drugs)

        drug_signatures = {}
        drug_metadata = {}

        self.model.eval()
        with torch.no_grad():
            for drug in tqdm(unique_drugs, desc="Processing drugs"):
                try:
                    # Get control samples for this drug
                    mask = (adata.obs[ptrb_col] == drug) & (adata.obs['condition'] == 'control')
                    if mask.sum() == 0:
                        continue

                    indices = np.where(mask)[0]

                    # Randomly select a few samples and average them
                    sample_indices = np.random.choice(indices, min(5, len(indices)), replace=False)

                    signatures = []
                    for idx in sample_indices:
                        sample = test_dataset[int(idx)]
                        batch = DataLoader([sample], batch_size=1, shuffle=False)
                        data = next(iter(batch)).to(self.device)

                        # Get model-predicted perturbation effect
                        pred, _ = self.model.predict(data)
                        ctrl_exp = data.x.cpu().numpy()

                        # Compute delta (perturbation effect)
                        delta = pred.cpu().numpy() - ctrl_exp
                        signatures.append(delta.flatten())

                    # Average the signatures
                    avg_signature = np.mean(signatures, axis=0)
                    drug_signatures[drug] = avg_signature

                    # Store metadata
                    drug_row = adata.obs[mask].iloc[0]
                    drug_name = smiles_to_name.get(drug, drug) if smiles_to_name else drug

                    drug_metadata[drug] = {
                        'drug_name': drug_name,
                        'cell_type': drug_row.get('cell_type', 'unknown'),
                        'smiles': drug_row.get('canonical_smiles', drug),
                        'n_samples': mask.sum()
                    }

                except Exception as e:
                    print(f"Failed to process drug {drug[:30]}: {e}")
                    continue

        print(f"Extracted signatures for {len(drug_signatures)} drugs")

        # Save drug metadata to CSV
        metadata_df = pd.DataFrame.from_dict(drug_metadata, orient='index')
        metadata_df.index.name = 'drug_id'
        metadata_df.reset_index(inplace=True)
        metadata_path = os.path.join(self.output_dir, 'drug_metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Drug metadata saved to {metadata_path}")

        # Store metadata for later use
        self.drug_metadata = drug_metadata

        return drug_signatures, drug_metadata

    def compute_drug_similarity(self, drug_signatures, save_to_csv=True):
        """
        Compute the pairwise drug similarity matrix.
        """
        print("Computing drug-drug similarity...")

        drugs = list(drug_signatures.keys())
        signatures = np.array([drug_signatures[d] for d in drugs])

        # Standardize
        scaler = StandardScaler()
        signatures_scaled = scaler.fit_transform(signatures)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(signatures_scaled)

        similarity_df = pd.DataFrame(similarity_matrix, index=drugs, columns=drugs)

        # Save similarity matrix to CSV
        if save_to_csv:
            similarity_path = os.path.join(self.output_dir, 'drug_similarity_matrix.csv')
            similarity_df.to_csv(similarity_path)
            print(f"Drug similarity matrix saved to {similarity_path}")

        return similarity_df

    def find_similar_drugs(self, target_drug, similarity_df, drug_metadata=None, top_k=10, save_to_csv=True):
        """
        Find the drugs most similar to a target drug.

        Args:
            target_drug: Target drug ID (SMILES or other identifier)
            similarity_df: Drug similarity matrix
            drug_metadata: Drug metadata dictionary (optional; falls back to self.drug_metadata
                           or the drug ID as the name if not provided)
            top_k: Number of top similar drugs to return
            save_to_csv: Whether to save results to CSV
        """
        if target_drug not in similarity_df.index:
            print(f"Drug {target_drug} not found")
            return None

        # If drug_metadata is not provided, try using the stored version
        if drug_metadata is None:
            if self.drug_metadata is not None:
                drug_metadata = self.drug_metadata
            else:
                drug_metadata = {drug: {'drug_name': drug} for drug in similarity_df.index}

        similarities = similarity_df.loc[target_drug].sort_values(ascending=False)
        # Exclude self
        similarities = similarities[similarities.index != target_drug]

        top_similar = similarities.head(top_k)

        # Save to CSV (including drug names)
        if save_to_csv:
            target_name = drug_metadata.get(target_drug, {}).get('drug_name', target_drug)

            result_df = pd.DataFrame({
                'target_drug_id': target_drug,
                'target_drug_name': target_name,
                'similar_drug_id': top_similar.index,
                'similar_drug_name': [drug_metadata.get(d, {}).get('drug_name', d) for d in top_similar.index],
                'similarity_score': top_similar.values
            })

            # Use a safe filename (using drug name)
            safe_drug_name = target_name[:50].replace('/', '_').replace('\\', '_').replace(' ', '_')
            similar_path = os.path.join(self.output_dir, f'similar_drugs_{safe_drug_name}.csv')
            result_df.to_csv(similar_path, index=False)
            print(f"Similar drugs for {target_name[:30]} saved to {similar_path}")

        return top_similar

    def predict_repurposing_candidates(self, disease_drugs, all_drugs_signatures,
                                      drug_metadata=None, similarity_threshold=0.7):
        """
        Predict potential repurposing candidates based on known disease drugs.

        Args:
            disease_drugs: List of drug IDs known to treat a given disease
            all_drugs_signatures: Dictionary of signatures for all drugs
            drug_metadata: Drug metadata dictionary (optional; falls back to self.drug_metadata
                           or drug IDs as names if not provided)
            similarity_threshold: Similarity threshold for candidate selection
        """
        print(f"Predicting repurposing candidates for {len(disease_drugs)} known drugs...")

        # If drug_metadata is not provided, try using the stored version
        if drug_metadata is None:
            if self.drug_metadata is not None:
                drug_metadata = self.drug_metadata
                print("Using stored drug_metadata from extract_drug_signatures()")
            else:
                print("Warning: No drug_metadata provided, using drug IDs as names")
                drug_metadata = {drug: {'drug_name': drug} for drug in all_drugs_signatures.keys()}

        # Compute similarity matrix
        similarity_df = self.compute_drug_similarity(all_drugs_signatures, save_to_csv=True)

        # Collect candidate drugs
        candidates = defaultdict(list)

        for known_drug in disease_drugs:
            if known_drug not in similarity_df.index:
                continue

            similar_drugs = self.find_similar_drugs(known_drug, similarity_df, drug_metadata,
                                                   top_k=20, save_to_csv=False)

            for candidate_drug, similarity in similar_drugs.items():
                if similarity >= similarity_threshold:
                    candidates[candidate_drug].append({
                        'reference_drug': known_drug,
                        'reference_drug_name': drug_metadata.get(known_drug, {}).get('drug_name', known_drug),
                        'similarity': similarity
                    })

        # Summarize results
        repurposing_results = []
        for candidate, references in candidates.items():
            avg_similarity = np.mean([r['similarity'] for r in references])
            candidate_name = drug_metadata.get(candidate, {}).get('drug_name', candidate)

            repurposing_results.append({
                'candidate_drug_id': candidate,
                'candidate_drug_name': candidate_name,
                'n_similar_drugs': len(references),
                'avg_similarity': avg_similarity,
                'max_similarity': max([r['similarity'] for r in references]),
                'min_similarity': min([r['similarity'] for r in references]),
                'reference_drug_ids': '; '.join([r['reference_drug'] for r in references]),
                'reference_drug_names': '; '.join([r['reference_drug_name'] for r in references])
            })

        results_df = pd.DataFrame(repurposing_results)
        results_df = results_df.sort_values('avg_similarity', ascending=False)

        # Save repurposing prediction results to CSV
        repurposing_path = os.path.join(self.output_dir, 'drug_repurposing_candidates.csv')
        results_df.to_csv(repurposing_path, index=False)
        print(f"Drug repurposing candidates saved to {repurposing_path}")

        # Save detailed similarity information
        detailed_results = []
        for candidate, references in candidates.items():
            candidate_name = drug_metadata.get(candidate, {}).get('drug_name', candidate)
            for ref in references:
                detailed_results.append({
                    'candidate_drug_id': candidate,
                    'candidate_drug_name': candidate_name,
                    'reference_drug_id': ref['reference_drug'],
                    'reference_drug_name': ref['reference_drug_name'],
                    'similarity': ref['similarity']
                })

        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = os.path.join(self.output_dir, 'drug_repurposing_detailed.csv')
        detailed_df.to_csv(detailed_path, index=False)
        print(f"Detailed repurposing results saved to {detailed_path}")

        return results_df

    def visualize_drug_space(self, drug_signatures, drug_metadata=None,
                            highlight_drugs=None, save_path=None):
        """
        Visualize the drug space using UMAP.

        Args:
            drug_signatures: Dictionary of drug signatures
            drug_metadata: Drug metadata dictionary (optional; falls back to self.drug_metadata
                           or drug IDs if not provided)
            highlight_drugs: List of drug IDs to highlight
            save_path: Path to save the figure
        """
        from umap import UMAP

        if save_path is None:
            save_path = os.path.join(self.output_dir, 'drug_space_umap.pdf')

        drugs = list(drug_signatures.keys())
        signatures = np.array([drug_signatures[d] for d in drugs])

        # If drug_metadata is not provided, try using the stored version
        if drug_metadata is None:
            if self.drug_metadata is not None:
                drug_metadata = self.drug_metadata
            else:
                drug_metadata = {
                    drug: {
                        'drug_name': drug,
                        'cell_type': 'unknown',
                        'smiles': drug
                    } for drug in drugs
                }

        # UMAP dimensionality reduction
        print("Performing UMAP dimensionality reduction...")
        reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding = reducer.fit_transform(signatures)

        # Save UMAP coordinates to CSV (including drug names)
        umap_df = pd.DataFrame({
            'drug_id': drugs,
            'drug_name': [drug_metadata.get(d, {}).get('drug_name', d) for d in drugs],
            'UMAP1': embedding[:, 0],
            'UMAP2': embedding[:, 1],
            'cell_type': [drug_metadata.get(d, {}).get('cell_type', 'unknown') for d in drugs],
            'smiles': [drug_metadata.get(d, {}).get('smiles', '') for d in drugs]
        })
        umap_path = os.path.join(self.output_dir, 'drug_space_umap_coordinates.csv')
        umap_df.to_csv(umap_path, index=False)
        print(f"UMAP coordinates saved to {umap_path}")

        # Visualization
        plt.figure(figsize=(12, 10))

        # Color by cell type
        cell_types = [drug_metadata.get(d, {}).get('cell_type', 'unknown') for d in drugs]
        unique_cell_types = list(set(cell_types))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_cell_types)))
        color_map = {ct: colors[i] for i, ct in enumerate(unique_cell_types)}

        for ct in unique_cell_types:
            mask = np.array([c == ct for c in cell_types])
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[color_map[ct]], label=ct, alpha=0.6, s=50)

        # Highlight specific drugs (annotated with drug names)
        if highlight_drugs:
            for drug in highlight_drugs:
                if drug in drugs:
                    idx = drugs.index(drug)
                    drug_name = drug_metadata.get(drug, {}).get('drug_name', drug)
                    plt.scatter(embedding[idx, 0], embedding[idx, 1],
                              c='red', s=200, marker='*', edgecolors='black', linewidths=2)
                    plt.annotate(drug_name[:20], (embedding[idx, 0], embedding[idx, 1]),
                               fontsize=8, ha='center',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.title('Drug Perturbation Space (UMAP)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Drug space visualization saved to {save_path}")

        return umap_df


