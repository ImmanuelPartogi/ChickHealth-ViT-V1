import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import random
from torchvision import transforms

# Default class names
DEFAULT_CLASSES = ["Chicken_Coccidiosis", "Chicken_Healthy", "Chicken_NewCastleDisease", "Chicken_Salmonella"]

# Image size - for Vision Transformer we need 224x224
IMAGE_SIZE = 224

class ChickenFecesDataset(Dataset):
    """
    Dataset for chicken feces classification.
    Simplified implementation that handles train/test split structure.
    """
    def __init__(self, data_dir, transform=None, classes=DEFAULT_CLASSES):
        """
        Initialize dataset
        
        Args:
            data_dir (str): Path to dataset directory
            transform: Transformations to apply to images
            classes (list): List of class names
        """
        self.data_dir = data_dir
        self.transform = transform
        self.classes = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        
        # Search for all image files in data directory
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                class_files = [f for f in os.listdir(class_dir) 
                              if os.path.isfile(os.path.join(class_dir, f)) and 
                              f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                print(f"Class {class_name}: {len(class_files)} images found")
                
                for img_name in class_files:
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        if not self.samples:
            raise RuntimeError(f"No images found in {data_dir}. Please check the folder structure.")
            
        print(f"Total dataset: {len(self.samples)} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Open image and convert to RGB
            image = Image.open(img_path).convert('RGB')
            
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)
            
            return image, label
                
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return placeholder image if error
            dummy_img = torch.zeros((3, IMAGE_SIZE, IMAGE_SIZE))
            return dummy_img, label

def get_transforms(train=True, img_size=IMAGE_SIZE):
    """
    Get image transforms for training or validation.
    Simplified but effective augmentation strategy.
    
    Args:
        train (bool): Whether to return transforms for training or validation
        img_size (int): Size of the image
    
    Returns:
        transforms.Compose: Composed transforms
    """
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if train:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Validation transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def create_dataloaders(data_dir, batch_size=16, train_ratio=0.8, num_workers=4, classes=DEFAULT_CLASSES):
    """
    Create and return DataLoaders for training and validation.
    Handles both train/test folder structure and single folder structure.
    
    Args:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size for DataLoader
        train_ratio (float): Ratio of data to use for training
        num_workers (int): Number of worker processes for data loading
        classes (list): List of class names
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    print(f"Creating dataset from directory: {data_dir}")
    
    # Check if train/test structure exists
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print(f"Found train/test directories structure.")
        
        # Create train dataset
        train_dataset = ChickenFecesDataset(
            train_dir,
            transform=get_transforms(train=True),
            classes=classes
        )
        
        # Create test/validation dataset
        val_dataset = ChickenFecesDataset(
            test_dir,
            transform=get_transforms(train=False),
            classes=classes
        )
    else:
        print(f"No train/test structure found. Using train_ratio={train_ratio} to split dataset.")
        
        # Create full dataset
        full_dataset = ChickenFecesDataset(
            data_dir,
            transform=None,  # No transform yet
            classes=classes
        )
        
        # Calculate split sizes
        dataset_size = len(full_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = dataset_size - train_size
        
        # Split dataset
        indices = list(range(dataset_size))
        random.shuffle(indices)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets with appropriate transforms
        train_dataset = ChickenFecesDataset(
            data_dir, 
            transform=get_transforms(train=True),
            classes=classes
        )
        train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
        
        val_dataset = ChickenFecesDataset(
            data_dir, 
            transform=get_transforms(train=False),
            classes=classes
        )
        val_dataset.samples = [full_dataset.samples[i] for i in val_indices]
        
        print(f"Split dataset: {len(train_dataset)} for training, {len(val_dataset)} for validation")
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def calculate_class_weights(data_loader):
    """
    Calculate class weights to handle imbalanced classes
    
    Args:
        data_loader: DataLoader for the dataset
        
    Returns:
        torch.Tensor: Class weights
    """
    class_counts = torch.zeros(len(DEFAULT_CLASSES))
    
    # Count samples per class
    for _, labels in data_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Calculate weights (inverse of frequency)
    weights = 1.0 / class_counts
    # Normalize weights
    weights = weights / weights.sum() * len(DEFAULT_CLASSES)
    
    return weights