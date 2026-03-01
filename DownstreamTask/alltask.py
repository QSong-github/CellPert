# Main execution script
import torch
from model import GINVAE
from dataset import ChunkedGeneGraphDataset
import scanpy as sc
import os
from task1 import CrossDatasetConsistencyAnalysis
from task2 import DrugDiseaseGeneNetwork
from task3 import DrugRepurposing
from task4 import DrugSynergyPrediction
from task5 import DiseaseGenePrioritization
from task6 import PerturbationSensitivityNetwork

def run_all_downstream_tasks(task_list):
    """
    Example of running all downstream tasks
    """
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GINVAE(input_dim=1, hidden_dim=300, latent_dim=100, num_layers=2).to(device)
    model.load_state_dict(torch.load('../src/best_gin_vae_model_node_level.pth'))
    model.eval()
    
    # Load data
    lincs_path = "/blue/qsong1/wang.qing/scPerturb/scUP/scGP/lincs/merged_all_965_with_morgan.h5ad"
    tahoe_paths = [f"/blue/qsong1/wang.qing/scPerturb/scUP/scGP/minitahoe/p{i}_with_morgan.h5ad" for i in range(1, 15)]
    
    lincs_adata = sc.read_h5ad(lincs_path)
    tahoe_adatas = [sc.read_h5ad(p) for p in tahoe_paths[:]]  
    
    # Create test datasets
    lincs_test = ChunkedGeneGraphDataset([lincs_path], split='test')
    tahoe_test = [ChunkedGeneGraphDataset([p], split='test') for p in tahoe_paths[:]]
    
    # Process reference dataset
    # if os.path.exists('./output/reference_latents.pkl'):
    #     print("\n=== Loading Cached Reference Latents ===")
    # else:
    print("\n=== Processing Reference Dataset ===")
    model.process_reference_dataset(
        x_ref_dataset=lincs_test,
        save_path='./output/reference_latents.pkl',
        force_reprocess=True
    )
    

    # Task 1: Cross-dataset validation
    print("\n=== Task 1: Cross-Dataset Consistency Analysis (LINCS vs Tahoe) ===")
    if "Cross-Dataset Consistency Analysis" in task_list:
        task1 = CrossDatasetConsistencyAnalysis(model, device)

        metrics = task1.comprehensive_analysis(
            lincs_adata=lincs_adata,
            tahoe_adatas=tahoe_adatas,
            lincs_test=lincs_test,
            tahoe_tests=tahoe_test,
            save_dir='./output/task1_cross_dataset'
        )

        print("\n✓ Task 1 completed!")
    
    # Task 2: Drug-disease gene network
    if "Drug-Disease Gene Network" in task_list:
        print("\n=== Task 2: Drug-Disease Gene Network ===")
        task2 = DrugDiseaseGeneNetwork(model, device)
        
        # Method 1: Analyze a single drug
        drug_id = lincs_adata.obs['canonical_smiles'].unique()[0]
        drug_analysis = task2.analyze_drug_perturbation(lincs_adata, drug_id)
        
        if drug_analysis and drug_analysis['network'] is not None:
            task2.visualize_network(
                drug_analysis['network'],
                drug_analysis['hub_genes'],
                drug_analysis['deg_genes'],
                drug_analysis['log2fc']
            )
        
        # Method 2: Batch analyze multiple drugs (recommended)
        print("\n--- Batch analyzing multiple drugs ---")
        summary_df = task2.batch_analyze_drugs(
            lincs_adata,
            n_drugs=5,  # Analyze 5 drugs
            save_dir='./output/task2_drug_analysis'
        )
        print("\nBatch Analysis Summary:")
        print(summary_df)
    
    # Task 3: Drug repurposing
    if "Drug Repurposing" in task_list:
        print("\n=== Task 3: Drug Repurposing ===")
        task3 = DrugRepurposing(model, device)
        drug_signatures, drug_metadata = task3.extract_drug_signatures(
            lincs_adata, lincs_test, max_drugs=50
        )
        similarity_df = task3.compute_drug_similarity(drug_signatures)
        
        # Assume some drugs are known for a disease
        known_disease_drugs = list(drug_signatures.keys())[:5]
        repurposing_candidates = task3.predict_repurposing_candidates(
            known_disease_drugs, drug_signatures
        )
        print("\nTop Repurposing Candidates:")
        print(repurposing_candidates.head(10))
        
        task3.visualize_drug_space(drug_signatures, drug_metadata, known_disease_drugs)
    
    # Task 4: Drug synergy
    if "Drug Synergy Prediction" in task_list:
        print("\n=== Task 4: Drug Synergy Prediction ===")
        task4 = DrugSynergyPrediction(model, device)
        synergy_results = task4.screen_drug_combinations(
            lincs_adata, lincs_test, n_combinations=30
        )
        task4.visualize_synergy_landscape(synergy_results)
        print("\nTop Synergistic Combinations:")
        print(synergy_results.head(10))
    
    # Task 5: Disease gene prioritization
    if "Disease Gene Prioritization" in task_list:  
        print("\n" + "="*70)
        print("TASK 5: Disease Gene Prioritization")
        print("="*70)
        task5 = DiseaseGenePrioritization(model, device)

        # Method 1: Analyze a single drug
        disease_drug = lincs_adata.obs['canonical_smiles'].unique()[0]
        gene_priority, gene_network = task5.prioritize_disease_genes(
            lincs_adata, disease_drug, lincs_test, top_k=50
        )

        if gene_priority is not None:
            task5.visualize_gene_prioritization(
                gene_priority, gene_network,
                save_path='./output/task5_gene_priority.png'
            )
            print("\nTop 10 Priority Genes:")
            print(gene_priority.head(10))

        # Method 2: Batch analysis (recommended)
        summary = task5.batch_prioritize_drugs(
            lincs_adata, 
            n_drugs=5,
            save_dir='./output/task5_gene_priority'
        )
        print("\nBatch Analysis Summary:")
        print(summary)

        print("\n✓ Task 5 completed!")
    
    # Task 6: Perturbation sensitivity network
    if "Perturbation Sensitivity Network" in task_list:
        print("\n=== Task 6: Perturbation Sensitivity Network ===")
        task6 = PerturbationSensitivityNetwork(model, device)
        sensitivity_results = task6.perform_complete_analysis(
            lincs_adata, lincs_test, n_perturbations=20, n_clusters=5
        )
        
        print("\n=== All Tasks Completed ===")

if __name__ == "__main__":
    # task_list = [
    #     "Cross-Dataset Consistency Analysis",
    #     "Drug-Disease Gene Network",        
    #     "Drug Repurposing",
    #     "Drug Synergy Prediction",
    #     "Disease Gene Prioritization",
    #     "Perturbation Sensitivity Network"
    # ]
    task_list = [
    "Drug Repurposing",
    ]
    run_all_downstream_tasks(task_list)