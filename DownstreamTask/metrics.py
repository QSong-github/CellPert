import torch
import numpy as np
import torch.nn.functional as F


def evaluate_prediction_samplewise_variable_length(recon_samples, x_samples, topk=100, device='cuda'):
    """
    Compute metrics for samples of variable length.

    Args:
        recon_samples: list of numpy arrays, each array is the reconstruction of one sample
        x_samples: list of numpy arrays, each array is the original data of one sample
        topk: k value for per-sample top-k gene overlap
        device: compute device
    """
    
    if len(recon_samples) != len(x_samples):
        raise ValueError("Number of reconstructed and original samples must match")
    
    # Store per-sample metrics
    sample_metrics = {
        'MSE': [],
        'MAE': [],
        'R2': [],
        'Pearson': [],
        'Spearman': [],
        'CosineSimilarity': [],
        'JS_Divergence': [],
        'SSIM': [],
        'TopK_Overlap': []  # Per-sample top-k overlap
    }
    
    with torch.no_grad():
        for pred_sample, true_sample in zip(recon_samples, x_samples):
            # Skip empty samples
            if len(pred_sample) == 0 or len(true_sample) == 0:
                continue
            
            # Ensure consistent length
            if len(pred_sample) != len(true_sample):
                min_len = min(len(pred_sample), len(true_sample))
                pred_sample = pred_sample[:min_len]
                true_sample = true_sample[:min_len]
            
            # Convert to torch tensor
            pred_tensor = torch.from_numpy(pred_sample).float().to(device)
            true_tensor = torch.from_numpy(true_sample).float().to(device)
            # Clamp negative values to zero
            pred_tensor[pred_tensor < 0] = 0

            # Assertion check (disabled)
            # assert torch.all(true_tensor >= 0), f"true_tensor has {torch.sum(true_tensor < 0).item()} negative values"
            # Skip samples shorter than 2 (cannot compute correlation)
            if len(pred_tensor) < 2:
                continue
            
            # Compute basic metrics
            diff = pred_tensor - true_tensor
            mse = torch.mean(diff ** 2).item()
            mae = torch.mean(torch.abs(diff)).item()
            
            # R²
            true_mean = torch.mean(true_tensor)
            ss_res = torch.sum(diff ** 2)
            ss_tot = torch.sum((true_tensor - true_mean) ** 2)
            r2 = (1 - ss_res / torch.clamp(ss_tot, min=1e-8)).item()
            
            # Pearson correlation coefficient
            pred_std = torch.std(pred_tensor, unbiased=False)
            true_std = torch.std(true_tensor, unbiased=False)
            
            if pred_std > 1e-8 and true_std > 1e-8:
                pred_mean = torch.mean(pred_tensor)
                true_mean = torch.mean(true_tensor)
                pred_normalized = (pred_tensor - pred_mean) / pred_std
                true_normalized = (true_tensor - true_mean) / true_std
                pearson = torch.mean(pred_normalized * true_normalized).item()
            else:
                pearson = float('nan')
            
            # Cosine similarity
            pred_norm = torch.norm(pred_tensor)
            true_norm = torch.norm(true_tensor)
            if pred_norm > 1e-8 and true_norm > 1e-8:
                cosine_sim = (torch.dot(pred_tensor, true_tensor) / (pred_norm * true_norm)).item()
            else:
                cosine_sim = float('nan')
            
            # Spearman correlation coefficient
            if len(pred_tensor) >= 3:
                try:
                    pred_ranks = torch.argsort(torch.argsort(pred_tensor)).float()
                    true_ranks = torch.argsort(torch.argsort(true_tensor)).float()
                    
                    pred_ranks_std = torch.std(pred_ranks, unbiased=False)
                    true_ranks_std = torch.std(true_ranks, unbiased=False)
                    
                    if pred_ranks_std > 1e-8 and true_ranks_std > 1e-8:
                        pred_ranks_mean = torch.mean(pred_ranks)
                        true_ranks_mean = torch.mean(true_ranks)
                        pred_ranks_norm = (pred_ranks - pred_ranks_mean) / pred_ranks_std
                        true_ranks_norm = (true_ranks - true_ranks_mean) / true_ranks_std
                        spearman = torch.mean(pred_ranks_norm * true_ranks_norm).item()
                    else:
                        spearman = float('nan')
                except:
                    spearman = float('nan')
            else:
                spearman = float('nan')
            
            # JS divergence
            try:
                pred_shifted = pred_tensor - torch.min(pred_tensor)
                true_shifted = true_tensor - torch.min(true_tensor)
                
                pred_sum = torch.sum(pred_shifted)
                true_sum = torch.sum(true_shifted)
                
                if pred_sum > 1e-8 and true_sum > 1e-8:
                    pred_prob = pred_shifted / pred_sum
                    true_prob = true_shifted / true_sum
                    
                    m = (pred_prob + true_prob) / 2
                    
                    eps = 1e-8
                    pred_prob = torch.clamp(pred_prob, min=eps)
                    true_prob = torch.clamp(true_prob, min=eps)
                    m = torch.clamp(m, min=eps)
                    
                    kl_pm = torch.sum(pred_prob * torch.log(pred_prob / m))
                    kl_tm = torch.sum(true_prob * torch.log(true_prob / m))
                    
                    js_divergence = (0.5 * (kl_pm + kl_tm)).item()
                else:
                    js_divergence = float('nan')
            except:
                js_divergence = float('nan')
            
            # SSIM
            try:
                C1 = 0.01 ** 2
                C2 = 0.03 ** 2
                
                pred_mean = torch.mean(pred_tensor)
                true_mean = torch.mean(true_tensor)
                pred_var = torch.var(pred_tensor, unbiased=False)
                true_var = torch.var(true_tensor, unbiased=False)
                covar = torch.mean((pred_tensor - pred_mean) * (true_tensor - true_mean))
                
                numerator1 = 2 * pred_mean * true_mean + C1
                numerator2 = 2 * covar + C2
                denominator1 = pred_mean ** 2 + true_mean ** 2 + C1
                denominator2 = pred_var + true_var + C2
                
                if denominator1 * denominator2 > 1e-8:
                    ssim = (numerator1 * numerator2 / (denominator1 * denominator2)).item()
                else:
                    ssim = float('nan')
            except:
                ssim = float('nan')
            
            # Per-sample top-k overlap
            try:
                sample_topk = min(topk, len(pred_tensor))
                if sample_topk > 0:
                    # Find top-k genes with largest absolute true values
                    true_abs = torch.abs(true_tensor)
                    top_true_idx = torch.topk(true_abs, sample_topk)[1]
                    
                    # Find top-k genes with largest prediction error
                    pred_error = torch.abs(pred_tensor - true_tensor)
                    top_pred_idx = torch.topk(pred_error, sample_topk)[1]
                    
                    # Compute intersection
                    top_true_set = set(top_true_idx.cpu().numpy())
                    top_pred_set = set(top_pred_idx.cpu().numpy())
                    
                    overlap = len(top_true_set & top_pred_set) / sample_topk
                else:
                    overlap = float('nan')
            except:
                overlap = float('nan')
            
            # Store metrics
            sample_metrics['MSE'].append(mse)
            sample_metrics['MAE'].append(mae)
            sample_metrics['R2'].append(r2)
            sample_metrics['Pearson'].append(pearson)
            sample_metrics['Spearman'].append(spearman)
            sample_metrics['CosineSimilarity'].append(cosine_sim)
            sample_metrics['JS_Divergence'].append(js_divergence)
            sample_metrics['SSIM'].append(ssim)
            sample_metrics['TopK_Overlap'].append(overlap)
    
    # Compute summary statistics
    def safe_mean_std(values):
        valid_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
        if len(valid_values) > 0:
            return np.mean(valid_values), np.std(valid_values), len(valid_values)
        else:
            return float('nan'), float('nan'), 0
    
    # Aggregate results
    results = {}
    for metric, values in sample_metrics.items():
        mean_val, std_val, count = safe_mean_std(values)
        results[metric] = mean_val
        # results[f'{metric}_std'] = std_val
        # results[f'{metric}_count'] = count
    
    results['total_samples'] = len([v for v in sample_metrics['MSE'] if not np.isnan(v)])
    
    return results


def evaluate_prediction_samplewise_gpu_fast(pred_exp, true_exp, topk=100, device='cuda'):
    """
    Faster GPU version including JS divergence and SSIM metrics.
    """
    # Convert to torch tensor
    if isinstance(pred_exp, np.ndarray):
        pred_exp = torch.from_numpy(pred_exp).float()
    if isinstance(true_exp, np.ndarray):
        true_exp = torch.from_numpy(true_exp).float()
    
    pred_exp = pred_exp.to(device)
    true_exp = true_exp.to(device)
    
    # Use PyTorch built-ins for accelerated computation
    with torch.no_grad():  # Gradient computation not needed
        # MSE and MAE
        diff = pred_exp - true_exp
        mse = torch.mean(diff ** 2, dim=1)
        mae = torch.mean(torch.abs(diff), dim=1)
        
        # R²
        true_mean_sample = torch.mean(true_exp, dim=1, keepdim=True)
        ss_res = torch.sum(diff ** 2, dim=1)
        ss_tot = torch.sum((true_exp - true_mean_sample) ** 2, dim=1)
        r2 = 1 - ss_res / torch.clamp(ss_tot, min=1e-8)
        
        # Normalize for correlation computation
        pred_std = torch.std(pred_exp, dim=1, keepdim=True, unbiased=False)
        true_std = torch.std(true_exp, dim=1, keepdim=True, unbiased=False)
        pred_mean = torch.mean(pred_exp, dim=1, keepdim=True)
        true_mean = torch.mean(true_exp, dim=1, keepdim=True)
        
        pred_normalized = (pred_exp - pred_mean) / torch.clamp(pred_std, min=1e-8)
        true_normalized = (true_exp - true_mean) / torch.clamp(true_std, min=1e-8)
        
        # Pearson correlation coefficient
        pearson = torch.mean(pred_normalized * true_normalized, dim=1)

        # Cosine similarity
        pred_norm = torch.norm(pred_exp, dim=1)
        true_norm = torch.norm(true_exp, dim=1)
        cosine_sim = torch.sum(pred_exp * true_exp, dim=1) / torch.clamp(pred_norm * true_norm, min=1e-8)
        
        # Spearman correlation (simplified, using ranks)
        pred_ranks = torch.argsort(torch.argsort(pred_exp, dim=1), dim=1).float()
        true_ranks = torch.argsort(torch.argsort(true_exp, dim=1), dim=1).float()
        
        pred_ranks_std = torch.std(pred_ranks, dim=1, keepdim=True, unbiased=False)
        true_ranks_std = torch.std(true_ranks, dim=1, keepdim=True, unbiased=False)
        pred_ranks_mean = torch.mean(pred_ranks, dim=1, keepdim=True)
        true_ranks_mean = torch.mean(true_ranks, dim=1, keepdim=True)
        
        pred_ranks_norm = (pred_ranks - pred_ranks_mean) / torch.clamp(pred_ranks_std, min=1e-8)
        true_ranks_norm = (true_ranks - true_ranks_mean) / torch.clamp(true_ranks_std, min=1e-8)
        
        spearman = torch.mean(pred_ranks_norm * true_ranks_norm, dim=1)
        
        # JS divergence (optimized)
        # Convert data to probability distributions
        pred_shifted = pred_exp - torch.min(pred_exp, dim=1, keepdim=True)[0]
        true_shifted = true_exp - torch.min(true_exp, dim=1, keepdim=True)[0]

        # Normalize to probability distributions
        pred_prob = pred_shifted / torch.clamp(torch.sum(pred_shifted, dim=1, keepdim=True), min=1e-8)
        true_prob = true_shifted / torch.clamp(torch.sum(true_shifted, dim=1, keepdim=True), min=1e-8)

        # Compute midpoint distribution
        m = (pred_prob + true_prob) / 2

        # Add small epsilon to avoid log(0)
        eps = 1e-8
        pred_prob = torch.clamp(pred_prob, min=eps)
        true_prob = torch.clamp(true_prob, min=eps)
        m = torch.clamp(m, min=eps)

        # Compute KL divergence
        kl_pm = torch.sum(pred_prob * torch.log(pred_prob / m), dim=1)
        kl_tm = torch.sum(true_prob * torch.log(true_prob / m), dim=1)

        # JS divergence
        js_divergence = 0.5 * (kl_pm + kl_tm)

        # SSIM (optimized)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Compute variance and covariance
        pred_var = torch.var(pred_exp, dim=1, keepdim=True, unbiased=False)
        true_var = torch.var(true_exp, dim=1, keepdim=True, unbiased=False)
        covar = torch.mean((pred_exp - pred_mean) * (true_exp - true_mean), dim=1, keepdim=True)

        # Compute SSIM
        numerator1 = 2 * pred_mean * true_mean + C1
        numerator2 = 2 * covar + C2
        denominator1 = pred_mean ** 2 + true_mean ** 2 + C1
        denominator2 = pred_var + true_var + C2
        
        ssim = ((numerator1 * numerator2) / torch.clamp(denominator1 * denominator2, min=1e-8)).squeeze()
        
        # Top-k overlap
        pred_global_mean = torch.mean(pred_exp, dim=0)
        true_global_mean = torch.mean(true_exp, dim=0)
        
        true_abs = torch.abs(true_global_mean)
        top_true_idx = torch.topk(true_abs, topk)[1]
        
        diff_global = torch.abs(true_global_mean - pred_global_mean)
        top_pred_idx = torch.topk(diff_global, topk)[1]
        
        # Compute intersection - fully on GPU
        # Create boolean masks to find intersection
        mask_true = torch.zeros(pred_exp.shape[1], dtype=torch.bool, device=device)
        mask_pred = torch.zeros(pred_exp.shape[1], dtype=torch.bool, device=device)
        
        mask_true[top_true_idx] = True
        mask_pred[top_pred_idx] = True
        
        overlap = (mask_true & mask_pred).sum().float().item() / topk
    
    # Handle NaN values and return results
    def safe_mean(tensor):
        valid_mask = ~torch.isnan(tensor)
        if valid_mask.sum() > 0:
            return tensor[valid_mask].mean().cpu().item()
        else:
            return float('nan')
    
    return {
        'MSE': safe_mean(mse),
        'MAE': safe_mean(mae),
        'R2': safe_mean(r2),
        'Pearson': safe_mean(pearson),
        'Spearman': safe_mean(spearman),
        'CosineSimilarity': safe_mean(cosine_sim),
        'JS_Divergence': safe_mean(js_divergence),
        'SSIM': safe_mean(ssim),
        f'Top{topk}_Overlap': overlap
    }