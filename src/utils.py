
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch 1.8+
    torch.use_deterministic_algorithms(True, warn_only=True)

class BatchSizeWarmupScheduler:
    def __init__(self, optimizer, target_lr, warmup_epochs=5):
        self.optimizer = optimizer
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def step_epoch(self):
        if self.current_epoch < self.warmup_epochs:
            # warmup
            lr = self.target_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            lr = self.target_lr
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr
    
def process_variable_size_batches_with_mask(all_recon, all_x, all_masks):
    processed_samples_recon = []
    processed_samples_x = []
    
    for recon_batch, x_batch, mask_batch in zip(all_recon, all_x, all_masks):
        batch_size = recon_batch.size(0)
        
        for i in range(batch_size):
            sample_mask = mask_batch[i]  # [max_nodes] boolean mask
            valid_node_count = sample_mask.sum().item()
            
            if valid_node_count > 0:
                sample_recon = recon_batch[i][sample_mask]  # [valid_nodes]
                sample_x = x_batch[i][sample_mask]  # [valid_nodes]
                
                processed_samples_recon.append(sample_recon.numpy())
                processed_samples_x.append(sample_x.numpy())
    
    return processed_samples_recon, processed_samples_x