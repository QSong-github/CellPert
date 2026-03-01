import torch
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.data import DataLoader
import warnings
from tqdm import tqdm
import random
warnings.filterwarnings('ignore')


class DrugSynergyPrediction:

    """
    Task 4: Drug Synergy Prediction
    Predict the synergistic effect when two drugs are used in combination
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict_combination_effect(self, data, drug1_context, drug2_context):
        """
        Predict the combination effect of two drugs.
        Uses the Bliss independence model as a baseline.
        """
        self.model.eval()
        with torch.no_grad():
            # Encode the control sample
            node_mu, node_logvar, graph_mu, graph_logvar = self.model.encoder(
                data.x, data.edge_index, data.batch
            )

            ctrl_latent = self.model.reparameterize(graph_mu, graph_logvar)

            # Predict single-drug effects
            drug1_latent = ctrl_latent + drug1_context
            drug2_latent = ctrl_latent + drug2_context

            # Combination effect (additive model - Bliss independence)
            combo_latent_additive = ctrl_latent + drug1_context + drug2_context

            # Decode predictions
            pred_drug1, _ = self.model.decoder(drug1_latent, data.edge_index, data.batch)
            pred_drug2, _ = self.model.decoder(drug2_latent, data.edge_index, data.batch)
            pred_combo_add, _ = self.model.decoder(combo_latent_additive, data.edge_index, data.batch)

        return pred_drug1, pred_drug2, pred_combo_add

    def calculate_synergy_score(self, pred_drug1, pred_drug2, pred_combo, baseline):
        """
        Calculate the synergy score.

        Bliss Score = O - E
        where O = observed combination effect
              E = expected effect under independence = drug1 + drug2 - drug1*drug2

        Positive score = synergy
        Negative score = antagonism
        """
        # Normalize to 0-1 range
        def normalize(x, baseline):
            return (x - baseline) / (baseline + 1e-8)

        norm_d1 = normalize(pred_drug1, baseline)
        norm_d2 = normalize(pred_drug2, baseline)
        norm_combo = normalize(pred_combo, baseline)

        # Bliss expected value
        expected = norm_d1 + norm_d2 - norm_d1 * norm_d2

        # Synergy score
        synergy = norm_combo - expected

        return synergy.mean().item(), synergy

    def screen_drug_combinations(self, adata, test_dataset,
                                cell_type=None, n_combinations=50):
        """
        Screen drug combinations and evaluate synergistic effects.
        """
        print("Screening drug combinations for synergy...")

        # Get the list of available drugs
        ptrb_col = 'sub_ptrb' if 'sub_ptrb' in adata.obs.columns else 'canonical_smiles'
        drugs = adata.obs[ptrb_col].unique()

        if cell_type:
            drugs = adata.obs[adata.obs['cell_type'] == cell_type][ptrb_col].unique()

        # Limit number of drugs
        drugs = drugs[:min(20, len(drugs))]

        # Extract biological context for each drug
        print("Extracting drug contexts...")
        drug_contexts = {}

        for drug in tqdm(drugs, desc="Processing drugs"):
            try:
                # Get the reference samples for this drug
                mask = (adata.obs[ptrb_col] == drug) & (adata.obs['condition'] == 'control')
                if mask.sum() == 0:
                    continue

                indices = np.where(mask)[0][:3]  # Limit number of samples

                contexts = []
                for idx in indices:
                    sample = test_dataset[int(idx)]
                    batch = DataLoader([sample], batch_size=1, shuffle=False)
                    data = next(iter(batch)).to(self.device)

                    # Get biological context (assumes reference is already available)
                    if self.model._ctrl_ref_cache is None:
                        continue

                    # Encode
                    with torch.no_grad():
                        _, _, graph_mu, _ = self.model.encoder(
                            data.x, data.edge_index, data.batch
                        )
                        # Context = perturbed drug state - control
                        context = graph_mu - self.model._ctrl_ref_cache
                        contexts.append(context)

                if contexts:
                    drug_contexts[drug] = torch.mean(torch.cat(contexts, dim=0), dim=0, keepdim=True)

            except Exception as e:
                continue

        print(f"Extracted contexts for {len(drug_contexts)} drugs")

        # Evaluate drug combinations
        from itertools import combinations
        drug_pairs = list(combinations(drug_contexts.keys(), 2))

        if len(drug_pairs) > n_combinations:
            drug_pairs = random.sample(drug_pairs, n_combinations)

        results = []

        for drug1, drug2 in tqdm(drug_pairs, desc="Evaluating combinations"):
            try:
                # Get one test sample
                mask = adata.obs['condition'] == 'control'
                if cell_type:
                    mask = mask & (adata.obs['cell_type'] == cell_type)

                idx = np.where(mask)[0][0]
                sample = test_dataset[int(idx)]
                batch = DataLoader([sample], batch_size=1, shuffle=False)
                data = next(iter(batch)).to(self.device)

                # Predict combination effect
                pred_d1, pred_d2, pred_combo = self.predict_combination_effect(
                    data, drug_contexts[drug1], drug_contexts[drug2]
                )

                # Calculate synergy score
                baseline = data.x.cpu().numpy()
                synergy_score, _ = self.calculate_synergy_score(
                    pred_d1.cpu().numpy(),
                    pred_d2.cpu().numpy(),
                    pred_combo.cpu().numpy(),
                    baseline
                )

                results.append({
                    'drug1': drug1,
                    'drug2': drug2,
                    'synergy_score': synergy_score,
                    'effect_type': 'Synergy' if synergy_score > 0.1 else
                                  ('Antagonism' if synergy_score < -0.1 else 'Additive')
                })

            except Exception as e:
                continue

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('synergy_score', ascending=False)

        return results_df

    def visualize_synergy_landscape(self, results_df, top_k=20, save_path='synergy_landscape.png'):
        """
        Visualize the synergy landscape.
        """
        # Select top and bottom combinations
        top_synergy = results_df.head(top_k//2)
        top_antagonism = results_df.tail(top_k//2)
        plot_data = pd.concat([top_synergy, top_antagonism])

        plt.figure(figsize=(14, 8))

        # Create combination labels
        plot_data['combination'] = plot_data['drug1'].str[:10] + '\n+\n' + plot_data['drug2'].str[:10]

        # Color encoding
        colors = ['green' if x > 0.1 else 'red' if x < -0.1 else 'gray'
                 for x in plot_data['synergy_score']]

        plt.barh(range(len(plot_data)), plot_data['synergy_score'], color=colors, alpha=0.7)
        plt.yticks(range(len(plot_data)), plot_data['combination'], fontsize=8)
        plt.xlabel('Synergy Score (Bliss)', fontsize=12)
        plt.title('Drug Combination Synergy Landscape\n(Green=Synergy, Red=Antagonism)', fontsize=14)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Synergy landscape saved to {save_path}")
