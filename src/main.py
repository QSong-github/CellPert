import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm
from torch_geometric.data import DataLoader
import scanpy as sc
import numpy as np
import pandas as pd
from metrics import evaluate_prediction_samplewise_variable_length
import argparse
from tqdm import tqdm
import random
from utils import set_seed, process_variable_size_batches_with_mask
from dataset import ChunkedGeneGraphDataset, ChunkAwareRandomSampler
from torch_geometric.utils import to_dense_batch
from model import GINVAE, loss_function
import pickle
import os
from datetime import datetime


def create_condition_filtered_dataset(dataset, condition):
    """
    Create a filtered dataset containing only samples with specified condition.
    
    Args:
        dataset: Original dataset
        condition: 'control' or 'stimulated'
    
    Returns:
        List of filtered data samples
    """
    filtered_samples = []
    
    print(f"Filtering dataset for condition: {condition}")
    
    for i in tqdm(range(len(dataset)), desc=f"Filtering {condition} samples"):
        sample = dataset[i]
        if hasattr(sample, 'condition') and str(sample.condition) == condition:
            filtered_samples.append(sample)
    
    print(f"Found {len(filtered_samples)} samples with condition '{condition}'")
    return filtered_samples


def find_paired_samples(ctrl_samples, ptrb_samples, pairing_keys=['celltype', 'main_ptrb', 'sub_ptrb']):
    """
    Find paired control and stimulated samples based on metadata.
    
    Args:
        ctrl_samples: List of control samples
        ptrb_samples: List of stimulated samples  
        pairing_keys: Metadata keys to use for pairing
    
    Returns:
        List of (control_sample, stimulated_sample) pairs
    """
    print("Finding paired samples...")
    
    # Create lookup dictionaries
    ctrl_lookup = {}
    ptrb_lookup = {}
    
    # Build control lookup
    for sample in ctrl_samples:
        key_parts = []
        for key in pairing_keys:
            if hasattr(sample, key):
                key_parts.append(str(getattr(sample, key)))
            else:
                key_parts.append('unknown')
        lookup_key = '_'.join(key_parts)
        
        if lookup_key not in ctrl_lookup:
            ctrl_lookup[lookup_key] = []
        ctrl_lookup[lookup_key].append(sample)
    
    # Build stimulated lookup
    for sample in ptrb_samples:
        key_parts = []
        for key in pairing_keys:
            if hasattr(sample, key):
                key_parts.append(str(getattr(sample, key)))
            else:
                key_parts.append('unknown')
        lookup_key = '_'.join(key_parts)
        
        if lookup_key not in ptrb_lookup:
            ptrb_lookup[lookup_key] = []
        ptrb_lookup[lookup_key].append(sample)
    
    # Find pairs
    paired_samples = []
    common_keys = set(ctrl_lookup.keys()) & set(ptrb_lookup.keys())
    
    for key in common_keys:
        ctrl_list = ctrl_lookup[key]
        ptrb_list = ptrb_lookup[key]
        
        # Pair samples (take minimum to ensure pairing)
        min_samples = min(len(ctrl_list), len(ptrb_list))
        for i in range(min_samples):
            paired_samples.append((ctrl_list[i], ptrb_list[i]))
    
    print(f"Found {len(paired_samples)} paired samples")
    return paired_samples


def evaluate_perturbation_prediction(model, test_dataset, latent_ctrl_ref, latent_ptrb_ref, 
                                   args, max_test_pairs=None, compute_deg=True):
    """
    Evaluate perturbation prediction using paired control-stimulated samples.
    
    Args:
        model: Trained GINVAE model
        test_dataset: Test dataset containing both control and stimulated samples
        latent_ctrl_ref: Control reference latent [1, latent_dim]
        latent_ptrb_ref: Stimulated reference latent [1, latent_dim]
        args: Training arguments
        max_test_pairs: Maximum number of test pairs to evaluate
        compute_deg: Whether to compute DEG (delta difference) in addition to absolute values
    """
    model.eval()
    
    # Filter test dataset by condition
    print("Filtering test dataset by condition...")
    ctrl_test_samples = create_condition_filtered_dataset(test_dataset, 'control')
    ptrb_test_samples = create_condition_filtered_dataset(test_dataset, 'perturb')
    
    if len(ctrl_test_samples) == 0 or len(ptrb_test_samples) == 0:
        print("Error: No control or stimulated samples found in test dataset")
        return
    
    # Find paired samples
    paired_samples = find_paired_samples(ctrl_test_samples, ptrb_test_samples)
    print(f'Num of Test perturb:{len(paired_samples)}')
    
    if len(paired_samples) == 0:
        print("Error: No paired samples found")
        return
    
    # Limit number of test pairs for efficiency
    if max_test_pairs!=None:
        paired_samples = random.sample(paired_samples, max_test_pairs)
        print(f"Randomly selected {max_test_pairs} pairs for testing")
    
    print(f"Testing perturbation prediction on {len(paired_samples)} paired samples...")
    
    all_predictions = []
    all_ground_truth = []
    all_ctrl_values = []  # Store control values for DEG calculation
    
    with torch.no_grad():
        # Process pairs in batches
        batch_size = args.batch_size
        num_batches = (len(paired_samples) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Predicting perturbations"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(paired_samples))
            batch_pairs = paired_samples[start_idx:end_idx]
            
            # Separate control inputs and stimulated targets
            ctrl_batch_data = []
            ptrb_batch_data = []
            
            for ctrl_sample, ptrb_sample in batch_pairs:
                ctrl_batch_data.append(ctrl_sample)
                ptrb_batch_data.append(ptrb_sample)
            
            # Create DataLoader for control samples (inputs)
            if len(ctrl_batch_data) > 0:
                ctrl_loader = DataLoader(ctrl_batch_data, batch_size=len(ctrl_batch_data), shuffle=False)
                ctrl_batch = next(iter(ctrl_loader)).to(args.device)
                
                # Predict perturbation effect on control samples
                try:
                    predicted_ptrb_exp, pred_mask = model.predict(ctrl_batch, latent_ctrl_ref, latent_ptrb_ref)
                    
                    # Process ground truth stimulated samples
                    ptrb_loader = DataLoader(ptrb_batch_data, batch_size=len(ptrb_batch_data), shuffle=False)
                    ptrb_batch = next(iter(ptrb_loader)).to(args.device)
                    
                    # Get ground truth in same format
                    ptrb_ground_truth = ptrb_batch.x
                    gt_dense, gt_mask = to_dense_batch(ptrb_ground_truth, ptrb_batch.batch)
                    
                    # Get control values for DEG calculation
                    ctrl_values = ctrl_batch.x
                    ctrl_dense, ctrl_mask = to_dense_batch(ctrl_values, ctrl_batch.batch)
                    
                    # Ensure consistent dimensions
                    if predicted_ptrb_exp.dim() == 3 and predicted_ptrb_exp.size(-1) == 1:
                        predicted_ptrb_exp = predicted_ptrb_exp.squeeze(-1)
                    if gt_dense.dim() == 3 and gt_dense.size(-1) == 1:
                        gt_dense = gt_dense.squeeze(-1)
                    if ctrl_dense.dim() == 3 and ctrl_dense.size(-1) == 1:
                        ctrl_dense = ctrl_dense.squeeze(-1)
                    
                    all_predictions.append(predicted_ptrb_exp.cpu())
                    all_ground_truth.append(gt_dense.cpu())
                    all_ctrl_values.append(ctrl_dense.cpu())
                    
                except Exception as e:
                    print(f"Warning: Failed to process batch {batch_idx}: {e}")
                    continue
    
    if len(all_predictions) == 0:
        print("Error: No successful predictions generated")
        return
    
    # Process all predictions and ground truth
    print("Processing prediction results...")
    
    # Create proper boolean masks for predictions and ground truth
    pred_masks = []
    gt_masks = []
    ctrl_masks = []
    
    for pred, gt, ctrl in zip(all_predictions, all_ground_truth, all_ctrl_values):
        # Create boolean masks - True where data is valid
        pred_mask = torch.ones(pred.shape[:2], dtype=torch.bool)  # [batch_size, max_nodes]
        gt_mask = torch.ones(gt.shape[:2], dtype=torch.bool)      # [batch_size, max_nodes]
        ctrl_mask = torch.ones(ctrl.shape[:2], dtype=torch.bool)  # [batch_size, max_nodes]
        
        pred_masks.append(pred_mask)
        gt_masks.append(gt_mask)
        ctrl_masks.append(ctrl_mask)
    
    pred_samples, gt_samples = process_variable_size_batches_with_mask(
        all_predictions, all_ground_truth, pred_masks
    )
    
    ctrl_samples, _ = process_variable_size_batches_with_mask(
        all_ctrl_values, all_ground_truth, ctrl_masks
    )
    
    print(f"Processed {len(pred_samples)} prediction pairs")
    
    # Evaluate perturbation prediction accuracy (absolute values)
    metrics_perturbation = evaluate_prediction_samplewise_variable_length(pred_samples, gt_samples)
    print("\nPerturbation Prediction Metrics (Control → Predicted Stimulated vs Ground Truth Stimulated):")
    print(pd.Series(metrics_perturbation))
    
    results = {
        "absolute_metrics": metrics_perturbation,
        "predictions": pred_samples,
        "ground_truth": gt_samples,
        "control_values": ctrl_samples
    }
    
    # Optionally compute DEG (delta difference) metrics
    if compute_deg:
        print("\nComputing DEG (delta difference) metrics...")
        
        # Calculate delta differences
        pred_delta_samples = []  # predicted_perturb - control
        gt_delta_samples = []    # ground_truth_perturb - control
        
        for pred, gt, ctrl in zip(pred_samples, gt_samples, ctrl_samples):
            pred_delta = pred - ctrl  # predicted perturbation effect
            gt_delta = gt - ctrl      # ground truth perturbation effect
            
            pred_delta_samples.append(pred_delta)
            gt_delta_samples.append(gt_delta)
        
        # Evaluate DEG prediction accuracy using same metrics
        metrics_deg = evaluate_prediction_samplewise_variable_length(pred_delta_samples, gt_delta_samples)
        print("\nDEG Prediction Metrics (Predicted Delta vs Ground Truth Delta):")
        print(pd.Series(metrics_deg))
        
        results["deg_metrics"] = metrics_deg
        results["pred_delta"] = pred_delta_samples
        results["gt_delta"] = gt_delta_samples
    
    # ==================== writing the results ====================
    # take the plate id
    plate_id = args.test_dataset_id if hasattr(args, 'test_dataset_id') else 'unknown'
    
    # 1. append the metrics to the CSV
    csv_file = os.path.join(args.output_dir, "all_plates_results.csv")
    
    # assemble the row
    if compute_deg:
        csv_data = {
            'plate_id': [plate_id],
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'num_samples': [len(pred_samples)]
        }
        # absolute metrics
        for key, value in metrics_perturbation.items():
            csv_data[f'absolute_{key}'] = [value]
        # delta metrics
        for key, value in metrics_deg.items():
            csv_data[f'deg_{key}'] = [value]
    else:
        csv_data = {
            'plate_id': [plate_id],
            'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'num_samples': [len(pred_samples)]
        }
        for key, value in metrics_perturbation.items():
            csv_data[key] = [value]
    
    df = pd.DataFrame(csv_data)
    
    # write the header only if the file is new
    file_exists = os.path.isfile(csv_file)
    
    # append
    df.to_csv(csv_file, mode='a', header=not file_exists, index=False)
    print(f"\n✓ Results appended to: {csv_file}")
    
    # 2. write predictions and ground truth to a pkl
    pkl_dir = os.path.join(args.output_dir, 'predictions')
    os.makedirs(pkl_dir, exist_ok=True)
    
    pkl_file = os.path.join(pkl_dir, f'plate_{plate_id}_predictions.pkl')
    
    # assemble what is written
    pkl_data = {
        'plate_id': plate_id,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': len(pred_samples),
        'predictions': pred_samples,
        'ground_truth': gt_samples,
        'control_values': ctrl_samples,
        'metrics': {
            'absolute': metrics_perturbation
        }
    }
    
    if compute_deg:
        pkl_data['pred_delta'] = pred_delta_samples
        pkl_data['gt_delta'] = gt_delta_samples
        pkl_data['metrics']['deg'] = metrics_deg
    
    # write the pkl
    with open(pkl_file, 'wb') as f:
        pickle.dump(pkl_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Predictions saved to: {pkl_file}")
    
    # 3. also write a per-plate CSV, kept for compatibility
    single_plate_csv = os.path.join(args.output_dir, f'plate_{plate_id}_results.csv')
    df.to_csv(single_plate_csv, index=False)
    print(f"✓ Single plate results saved to: {single_plate_csv}")
    
    return results


# training
def train_gin_vae(args):
    # the training set is loaded only when training or testing
    train_dataset = None
    
    if args.train_flag:
        # Build gene graph from training data
        print("Loading training(ref) data...")
        
        train_dataset = ChunkedGeneGraphDataset(
            h5ad_paths=args.train_data_path,
            split='train',
            chunk_size=10000,  # tune to the available memory
            auto_build_graph=True,
            species=9606,
            required_score=700
        )
        # usage
        sampler = ChunkAwareRandomSampler(
            train_dataset, 
            chunk_size=5000, 
            shuffle_chunks=True,      # shuffle across chunks
            shuffle_within_chunk=False # keep the order inside a chunk, which reduces IO
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
        
        # Get dimensions from first data sample
        sample_data = train_dataset[0]
        input_dim = sample_data.x.size(1)
        print(f"Detected input_dim: {input_dim}")
        
        if args.from_ckpt:
            model = GINVAE(input_dim, args.hidden_dim, args.latent_dim, args.num_layers).to(args.device)
            model.load_state_dict(torch.load(args.ckpt_path))
        else:
            model = GINVAE(input_dim, args.hidden_dim, args.latent_dim, args.num_layers).to(args.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            mse_loss = 0
            graph_kl_loss = 0
            total_samples = 0
            
            # keep each batch and its mask
            all_recon_batches = []
            all_x_batches = []
            all_mask_batches = []
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch") as pbar:
                for batch in pbar:
                    batch = batch.to(args.device)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    recon_batch, mask, node_mu, node_logvar, graph_mu, graph_logvar = model(batch)
                    bid = batch.batch
                    original_x = batch.x
                    
                    # dense batch and its mask
                    b_original_x, node_mask = to_dense_batch(original_x, bid)
                    
                    # make the dimensions agree
                    if recon_batch.dim() == 3 and recon_batch.size(-1) == 1:
                        recon_batch = recon_batch.squeeze(-1)
                    if b_original_x.dim() == 3 and b_original_x.size(-1) == 1:
                        b_original_x = b_original_x.squeeze(-1)
                    
                    # Loss calculation
                    loss, mse, graph_kld = loss_function(
                        recon_batch, b_original_x, node_mu, node_logvar, graph_mu, graph_logvar, args
                    )
                    
                    # keep this batch for the evaluation at the end
                    all_recon_batches.append(recon_batch.cpu().detach())
                    all_x_batches.append(b_original_x.cpu().detach())
                    all_mask_batches.append(node_mask.cpu().detach())
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    batch_size = batch.batch.max().item() + 1
                    total_samples += batch_size
                    
                    train_loss += loss.item()
                    mse_loss += mse.item()
                    graph_kl_loss += graph_kld.item()
                    
                    pbar.set_postfix({
                        "Loss": f"{train_loss / total_samples:.8f}",
                        "MSE": f"{mse_loss / total_samples:.8f}",
                        "GraphKL": f"{graph_kl_loss / total_samples:.8f}"
                    })
            
            # evaluation at the end of the epoch
            train_loss_ep = train_loss / total_samples
            if train_loss_ep < best_loss:
                best_loss = train_loss_ep
                best_epoch = epoch
                torch.save(model.state_dict(), args.ckpt_path)

            print(f"best loss: {best_loss:.4f}, best_epoch: {best_epoch}")

            # take long time！
            # if epoch==args.epochs-1:
            # recon_samples, x_samples = process_variable_size_batches_with_mask(
            #     all_recon_batches, all_x_batches, all_mask_batches
            # )
            
            # print(f"Processed {len(recon_samples)} samples for evaluation")
            
            # # Reconstruction metrics
            # metrics = evaluate_prediction_samplewise_variable_length(recon_samples, x_samples)
            # print("Training Reconstruction Metrics:")
            # print(pd.Series(metrics))
    
    # Testing
    if args.test_flag:
        # Build gene graph from testing data
        test_dataset = ChunkedGeneGraphDataset(
            h5ad_paths=args.test_data_path,
            split='test',
            chunk_size=10000,  # tune to the available memory
            auto_build_graph=True,
            species=9606,
            required_score=700
        )

        # Get dimensions for model initialization
        sample_data = test_dataset[0]
        input_dim = sample_data.x.size(1) #1
        
        model = GINVAE(input_dim, args.hidden_dim, args.latent_dim, args.num_layers).to(args.device)
        model.load_state_dict(torch.load(args.ckpt_path))
        model.eval()
        
        # Step 1: Process reference dataset (from training data) to get biological context
        print("\nProcessing reference dataset for biological context...")
        
        # load the training set only when it is needed
        if train_dataset is None:
            print("Loading training(ref) data for reference...")
            train_dataset = ChunkedGeneGraphDataset(
                h5ad_paths=args.train_data_path,
                split='train',
                chunk_size=10000,
                auto_build_graph=True,
                species=9606,
                required_score=700
            )
        
        X_ref_dataset = train_dataset
        
        latent_ctrl_ref, latent_ptrb_ref, info = model.process_reference_dataset(
            x_ref_dataset=X_ref_dataset,
            save_path=args.ref_save_path,
            group_type=None,
            group_name=None,
            force_reprocess=False  # skip the work if the file already exists
        )
        
        if latent_ctrl_ref is None or latent_ptrb_ref is None:
            print("Failed to process reference dataset - skipping perturbation prediction")
            return model if args.train_flag else None
        
        # Step 2: Test standard reconstruction on all test data
        print("\nTesting standard reconstruction...")
        sampler = ChunkAwareRandomSampler(
            test_dataset, 
            chunk_size=5000, 
            shuffle_chunks=False,  # keep the order, which makes the output easier to analyse
            shuffle_within_chunk=False
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
        
        with torch.no_grad():
            all_recon_test_batches = []
            all_x_test_batches = []
            all_mask_test_batches = []
            
            for batch in tqdm(test_loader, desc="Testing reconstruction"):
                batch = batch.to(args.device)
                
                # Forward pass
                recon_batch, mask, node_mu, node_logvar, graph_mu, graph_logvar = model(batch)
                
                original_x = batch.x
                b_original_x, node_mask = to_dense_batch(original_x, batch.batch)
                
                if recon_batch.dim() == 3 and recon_batch.size(-1) == 1:
                    recon_batch = recon_batch.squeeze(-1)
                if b_original_x.dim() == 3 and b_original_x.size(-1) == 1:
                    b_original_x = b_original_x.squeeze(-1)
                
                all_recon_test_batches.append(recon_batch.cpu().detach())
                all_x_test_batches.append(b_original_x.cpu().detach())
                all_mask_test_batches.append(node_mask.cpu().detach())
            
            # process the test data, keeping each sample intact
            recon_test_samples, x_test_samples = process_variable_size_batches_with_mask(
                all_recon_test_batches, all_x_test_batches, all_mask_test_batches
            )
            
            print(f"Test: Processed {len(recon_test_samples)} samples")
            
            # reconstruction metrics on the test set
            metrics_test = evaluate_prediction_samplewise_variable_length(recon_test_samples, x_test_samples)
            print("Test Reconstruction Metrics:")
            print(pd.Series(metrics_test))
        
        # Step 3: Test perturbation prediction using correct logic
        print("\n" + "="*50)
        print("PERTURBATION PREDICTION EVALUATION")
        print("="*50)
        
        perturbation_metrics = evaluate_perturbation_prediction(
            model, test_dataset, latent_ctrl_ref, latent_ptrb_ref, args, max_test_pairs=None
        )
    
    return model if args.train_flag else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for training GIN-VAE")
    parser.add_argument("--train_data_path", type=str, default=['./data/merged_all_965_with_morgan.h5ad'])
    parser.add_argument("--test_data_path", type=str, default='./data/minitahoe/p1_with_morgan.h5ad')
    parser.add_argument("--ref_save_path", type=str, default='./output/reference_latents.pkl')
    parser.add_argument("--output_dir", type=str, default='./output')

    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=300)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GIN layers in encoder and decoder")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--train_flag", action='store_true', help="Enable training mode")
    parser.add_argument("--from_ckpt", action='store_true', help="Load from checkpoint")
    parser.add_argument("--ckpt_path", type=str, default='./best_gin_vae_model_node_level.pth')
    parser.add_argument("--test_flag", action='store_true', help="Enable testing mode")
    parser.add_argument("--test_dataset_id", type=int, default=1, help="Test dataset plate ID")
    
    parser.add_argument("--graph_kl_weight", type=float, default=0.1, help="Weight for graph-level KL divergence")
    
    args = parser.parse_args()
    
    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("*"*50)
    print("Current Args:", args)
    print("*"*50)
    set_seed(args.seed)
    
    train_gin_vae(args)