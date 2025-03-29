"""
Utility functions for model training, evaluation, and visualization.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

def save_checkpoint(model, optimizer, epoch, accuracy, loss, filepath):
    """
    Save model checkpoint.
    
    Args:
        model: Pytorch model
        optimizer: Optimizer
        epoch: Current epoch
        accuracy: Validation accuracy
        loss: Validation loss
        filepath: Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'loss': loss
    }, filepath)


def load_checkpoint(model, optimizer, filepath, device):
    """
    Load model checkpoint.
    
    Args:
        model: Pytorch model
        optimizer: Optimizer
        filepath: Path to the checkpoint
        device: Device to load the model (cpu/cuda)
        
    Returns:
        model: Loaded model
        optimizer: Loaded optimizer
        epoch: Epoch from the checkpoint
        accuracy: Accuracy from the checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    accuracy = checkpoint['accuracy']
    
    return model, optimizer, epoch, accuracy


def visualize_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    Visualize training and validation curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_accs: List of training accuracies
        val_accs: List of validation accuracies
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training dan Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training dan Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)


def visualize_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Create and save confusion matrix visualization.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)


def compute_metrics(y_true, y_pred, class_names):
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        dict: Dictionary containing metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'per_class_precision': precision_score(y_true, y_pred, average=None),
        'per_class_recall': recall_score(y_true, y_pred, average=None),
        'per_class_f1': f1_score(y_true, y_pred, average=None)
    }
    
    # Add per-class metrics with class names
    metrics['per_class_metrics'] = {}
    for i, class_name in enumerate(class_names):
        metrics['per_class_metrics'][class_name] = {
            'precision': metrics['per_class_precision'][i],
            'recall': metrics['per_class_recall'][i],
            'f1': metrics['per_class_f1'][i]
        }
    
    # Remove the numpy arrays to make it easier to print/serialize
    del metrics['per_class_precision']
    del metrics['per_class_recall']
    del metrics['per_class_f1']
    
    return metrics


def predict_single_image(model, image_tensor, class_names, device):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained model
        image_tensor: Preprocessed image tensor
        class_names: List of class names
        device: Device to run inference on (cpu/cuda)
        
    Returns:
        dict: Dictionary containing prediction results
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        
    prediction = {
        'class': class_names[predicted_class],
        'confidence': float(probabilities[predicted_class]) * 100,
        'probabilities': {class_names[i]: float(prob) * 100 for i, prob in enumerate(probabilities)}
    }
    
    return prediction