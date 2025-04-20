import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from utils import calculate_gradient_norm

class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_model(self, 
                   optimizer: torch.optim.Optimizer,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   epochs: int = 10) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        lr_history = []
        grad_norm_history = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                loss.backward()
                
                # Track gradient norm before optimizer step
                grad_norm = calculate_gradient_norm(self.model)
                grad_norm_history.append(grad_norm)
                
                optimizer.step()
                
                # Track learning rate
                current_lr = optimizer.param_groups[0]['lr']
                lr_history.append(current_lr)
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * correct / total
            
            # Store metrics
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
        return {
            'train_loss': train_losses,
            'train_acc': train_accs,
            'lr_history': lr_history,
            'grad_norm_history': grad_norm_history
        }, {
            'val_loss': val_losses,
            'val_acc': val_accs
        }
    
    def plot_metrics(self, train_metrics: Dict[str, List[float]], val_metrics: Dict[str, List[float]]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(train_metrics['train_loss']) + 1)
        ax1.plot(epochs, train_metrics['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, val_metrics['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracies
        ax2.plot(epochs, train_metrics['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_metrics['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        return fig