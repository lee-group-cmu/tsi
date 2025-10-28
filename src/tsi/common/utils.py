import click
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
import json
import hashlib


def create_experiment_hash(config, experiment_id=None):
    if experiment_id is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create a config hash for uniqueness (focusing on training-specific params)
        training_config = {k: v for k, v in config.items() 
                          if k in ['hidden_dims', 'lr', 'batch_size', 'n_epochs', 'weight_decay', 'lambda_gp', 'dropout_rate']}
        config_str = json.dumps(training_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        # Create readable training params string
        training_params = []
        if 'hidden_dims' in config and config['hidden_dims']:
            layers_str = "x".join(map(str, config['hidden_dims']))
            training_params.append(f"h{layers_str}")
        if 'lr' in config:
            training_params.append(f"lr{config['lr']}")
        if 'batch_size' in config:
            training_params.append(f"bs{config['batch_size']}")
        if 'n_epochs' in config:
            training_params.append(f"ep{config['n_epochs']}")
        if 'weight_decay' in config:
            training_params.append(f"wd{config['weight_decay']}")
        if 'lambda_gp' in config:
            training_params.append(f"gp{config['lambda_gp']}")
        if 'dropout_rate' in config:
            training_params.append(f"dr{config['dropout_rate']}")
        
        training_str = "_".join(training_params) if training_params else "default"
        experiment_id = f"{timestamp}_{training_str}_{config_hash}"
        return experiment_id
    else:
        return experiment_id


class IntList(click.ParamType):
    name = 'intlist'
    
    def convert(self, value, param, ctx):
        try:
            return tuple(int(x) for x in value.split(','))
        except ValueError:
            self.fail(f'{value!r} is not a valid comma-separated list of integers', param, ctx)


class TrainingLogger:
    """Class to handle training loss logging and visualization"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = {
            'epoch_losses': [],
            'bce_losses': [],
            'gp_losses': [],
            'epoch_times': [],
            'learning_rates': [],
            'metadata': {}
        }
        os.makedirs(log_dir, exist_ok=True)
    
    def log_epoch(self, epoch, epoch_loss, bce_loss, gp_loss, epoch_time, lr=None):
        """Log epoch-level metrics"""
        self.losses['epoch_losses'].append(epoch_loss)
        self.losses['bce_losses'].append(bce_loss)
        self.losses['gp_losses'].append(gp_loss)
        self.losses['epoch_times'].append(epoch_time)
        if lr is not None:
            self.losses['learning_rates'].append(lr)
    
    def set_metadata(self, **kwargs):
        """Set training metadata"""
        self.losses['metadata'].update(kwargs)
        self.losses['metadata']['timestamp'] = datetime.now().isoformat()
    
    def save_losses(self, filename="training_losses.pkl"):
        """Save losses to pickle file"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.losses, f)
        print(f"✓ Training losses saved to {filepath}")
    
    def save_losses_csv(self, filename="training_losses.csv"):
        """Save epoch losses to CSV for easy analysis"""
        filepath = os.path.join(self.log_dir, filename)
        df = pd.DataFrame({
            'epoch': range(1, len(self.losses['epoch_losses']) + 1),
            'loss': self.losses['epoch_losses'],
            'bce_loss': self.losses['bce_losses'],
            'gp_loss': self.losses['gp_losses'],
            'epoch_time': self.losses['epoch_times']
        })
        if self.losses['learning_rates']:
            df['learning_rate'] = self.losses['learning_rates']
        
        df.to_csv(filepath, index=False)
        print(f"✓ Training losses saved to CSV: {filepath}")
    
    def plot_training_curves(self, filename="training_curves.png", show_batch_losses=False):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves', fontsize=16)
        
        # Epoch losses
        axes[0, 0].plot(range(1, len(self.losses['epoch_losses']) + 1), 
                       self.losses['epoch_losses'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss per Epoch')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log scale epoch losses
        axes[0, 1].semilogy(range(1, len(self.losses['epoch_losses']) + 1), 
                           self.losses['epoch_losses'], 'b-', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (log scale)')
        axes[0, 1].set_title('Training Loss per Epoch (Log Scale)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Batch losses for recent epochs (if requested and available)
        if show_batch_losses and self.losses['batch_losses']:
            # Show last few epochs of batch losses
            recent_epochs = min(5, len(self.losses['batch_losses']))
            batch_data = []
            epoch_labels = []
            
            for i in range(-recent_epochs, 0):
                epoch_batches = self.losses['batch_losses'][i]
                batch_data.extend(epoch_batches)
                epoch_labels.extend([len(self.losses['epoch_losses']) + i + 1] * len(epoch_batches))
            
            if batch_data:
                axes[1, 0].plot(batch_data, 'g-', alpha=0.7, linewidth=1)
                axes[1, 0].set_xlabel('Batch (Recent Epochs)')
                axes[1, 0].set_ylabel('Batch Loss')
                axes[1, 0].set_title(f'Batch Losses (Last {recent_epochs} Epochs)')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Training time per epoch
        if self.losses['epoch_times']:
            axes[1, 1].plot(range(1, len(self.losses['epoch_times']) + 1), 
                           self.losses['epoch_times'], 'r-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_title('Training Time per Epoch')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(self.log_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Training curves saved to {filepath}")
    
    def print_summary(self):
        """Print training summary"""
        if not self.losses['epoch_losses']:
            print("No training data recorded")
            return
        
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(f"Total epochs: {len(self.losses['epoch_losses'])}")
        print(f"Final loss: {self.losses['epoch_losses'][-1]:.6f}")
        print(f"Best loss: {min(self.losses['epoch_losses']):.6f}")
        print(f"Improvement: {self.losses['epoch_losses'][0] - self.losses['epoch_losses'][-1]:.6f}")
        
        if self.losses['epoch_times']:
            total_time = sum(self.losses['epoch_times'])
            avg_time = np.mean(self.losses['epoch_times'])
            print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Average time per epoch: {avg_time:.2f} seconds")
        
        print("=" * 50)
