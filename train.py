import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from model import create_model
from dataset import create_dataloaders, DEFAULT_CLASSES, calculate_class_weights

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed} for reproducibility")

class EarlyStopping:
    """Early stopping handler to prevent overfitting"""
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = 100 * correct / total
    
    return val_loss, val_acc, all_preds, all_labels

def plot_confusion_matrix(cm, classes, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Use nicer labels for display (without the "Chicken_" prefix)
    display_classes = [cls.replace('Chicken_', '') for cls in classes]
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=display_classes, 
        yticklabels=display_classes
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training history plot saved to {save_path}")

def train_model(model, train_loader, val_loader, num_epochs=20, device='cuda', 
                learning_rate=0.001, save_dir='.'):
    """Train and validate model with early stopping"""
    # Create models directory
    models_dir = os.path.join(save_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Use class weights to handle imbalanced data
    if torch.cuda.is_available():
        class_weights = calculate_class_weights(train_loader).to(device)
    else:
        class_weights = None
    
    # Loss function with class weights and label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5)
    
    # Initialize tracking variables
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(models_dir, 'best_vit_chicken_classifier.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved with accuracy: {best_val_acc:.2f}%")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, DEFAULT_CLASSES, os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Plot training history
    plot_training_history(history, os.path.join(save_dir, 'training_history.png'))
    
    return history, best_val_acc

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train chicken feces classifier')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Directory to save model and results')
    parser.add_argument('--model_type', type=str, default='vit_base_patch16_224',
                        help='Model type to use (default: vit_base_patch16_224)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Reduce workers if on CPU
    if device.type == 'cpu':
        args.num_workers = min(args.num_workers, 2)
        # Also reduce batch size
        args.batch_size = min(args.batch_size, 8)
        print(f"Using {args.num_workers} workers and batch size {args.batch_size} for CPU")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = create_model(
        model_type=args.model_type,
        num_classes=len(DEFAULT_CLASSES),
        pretrained=True
    )
    model = model.to(device)
    
    # Train model
    history, best_acc = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        device=device,
        learning_rate=args.learning_rate,
        save_dir=args.output_dir
    )
    
    print(f"Training completed with best validation accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()