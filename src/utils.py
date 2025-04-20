import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import os

def plot_training_curves(optimizers_results: Dict[str, Tuple[Dict[str, List[float]], Dict[str, List[float]]]]):
    """
    Plot training and validation curves for multiple optimizers
    Args:
        optimizers_results: Dictionary with optimizer names as keys and (train_metrics, val_metrics) as values
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    for optimizer_name, (train_metrics, val_metrics) in optimizers_results.items():
        # Loss curves
        ax1.plot(train_metrics['train_loss'], label=f'{optimizer_name} - Train')
        ax1.plot(val_metrics['val_loss'], label=f'{optimizer_name} - Val', linestyle='--')
        
        # Accuracy curves
        ax2.plot(train_metrics['train_acc'], label=f'{optimizer_name} - Train')
        ax2.plot(val_metrics['val_acc'], label=f'{optimizer_name} - Val', linestyle='--')
        
        # Learning rate analysis
        if 'lr_history' in train_metrics:
            ax3.plot(train_metrics['lr_history'], label=optimizer_name)
        
        # Gradient norm analysis
        if 'grad_norm_history' in train_metrics:
            ax4.plot(train_metrics['grad_norm_history'], label=optimizer_name)
    
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_title('Learning Rate Changes')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Learning Rate')
    ax3.legend()
    ax3.grid(True)
    
    ax4.set_title('Gradient Norm')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('L2 Norm')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    return fig

def save_results_to_csv(optimizers_results: Dict[str, Tuple[Dict[str, List[float]], Dict[str, List[float]]]], 
                       output_path: str = 'results/optimization_comparison.csv') -> pd.DataFrame:
    """
    Save training results to a CSV file
    """
    results_data = []
    for optimizer_name, (train_metrics, val_metrics) in optimizers_results.items():
        final_train_loss = train_metrics['train_loss'][-1]
        final_train_acc = train_metrics['train_acc'][-1]
        final_val_loss = val_metrics['val_loss'][-1]
        final_val_acc = val_metrics['val_acc'][-1]
        
        results_data.append({
            'Optimizer': optimizer_name,
            'Final Train Loss': final_train_loss,
            'Final Train Accuracy': final_train_acc,
            'Final Validation Loss': final_val_loss,
            'Final Validation Accuracy': final_val_acc,
            'Convergence Time (epochs)': len(train_metrics['train_loss'])
        })
    
    df = pd.DataFrame(results_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

def calculate_gradient_norm(model: torch.nn.Module) -> float:
    """
    Calculate the L2 norm of gradients for all parameters in the model
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return torch.sqrt(torch.tensor(total_norm)).item()

def compare_convergence_speed(optimizers_results: Dict[str, Tuple[Dict[str, List[float]], Dict[str, List[float]]]], 
                            target_accuracy: float = 95.0) -> Dict[str, int]:
    """
    Compare how many epochs each optimizer took to reach a target accuracy
    """
    convergence_epochs = {}
    for optimizer_name, (train_metrics, _) in optimizers_results.items():
        accuracies = train_metrics['train_acc']
        for epoch, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                convergence_epochs[optimizer_name] = epoch + 1
                break
        if optimizer_name not in convergence_epochs:
            convergence_epochs[optimizer_name] = float('inf')
    
    return convergence_epochs