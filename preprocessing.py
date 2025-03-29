"""
Data preprocessing utilities for chicken feces classification.
"""

import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTFeatureExtractor
import numpy as np
from config import MODEL_NAME, IMAGE_SIZE, BATCH_SIZE, DATA_DIR

# Custom Dataset class for chicken feces images
class ChickenFecesDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir (str): Root directory of the dataset
            split (str): 'train' or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = os.listdir(self.root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms():
    """
    Get transforms for train and test datasets.
    
    Returns:
        tuple: (train_transform, test_transform)
    """
    # Feature extractor for ViT
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    
    # Transformasi untuk data training dengan augmentasi
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    
    # Transformasi untuk data testing tanpa augmentasi
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    
    return train_transform, test_transform


def preprocess_single_image(image):
    """
    Preprocess a single image for inference.
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    
    return transform(image).unsqueeze(0)


def create_data_loaders(batch_size=BATCH_SIZE, num_workers=4):
    """
    Create data loaders for training and testing.
    
    Args:
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: (train_loader, test_loader, class_names)
    """
    train_transform, test_transform = get_transforms()
    
    # Initialize datasets
    train_dataset = ChickenFecesDataset(
        root_dir=DATA_DIR, 
        split="train", 
        transform=train_transform
    )
    
    test_dataset = ChickenFecesDataset(
        root_dir=DATA_DIR, 
        split="test", 
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class names from the dataset
    class_names = list(train_dataset.class_to_idx.keys())
    
    return train_loader, test_loader, class_names


def analyze_dataset():
    """
    Analyze the dataset and print statistics.
    
    Returns:
        dict: Statistics about the dataset
    """
    train_dir = os.path.join(DATA_DIR, "train")
    test_dir = os.path.join(DATA_DIR, "test")
    
    class_names = os.listdir(train_dir)
    
    stats = {
        "total": {"train": 0, "test": 0},
        "classes": {}
    }
    
    # Count samples per class
    for class_name in class_names:
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        train_count = len(os.listdir(train_class_dir))
        test_count = len(os.listdir(test_class_dir))
        
        stats["classes"][class_name] = {
            "train": train_count,
            "test": test_count,
            "total": train_count + test_count
        }
        
        stats["total"]["train"] += train_count
        stats["total"]["test"] += test_count
    
    stats["total"]["all"] = stats["total"]["train"] + stats["total"]["test"]
    
    # Print statistics
    print(f"Dataset Statistics:")
    print(f"Total samples: {stats['total']['all']}")
    print(f"Training samples: {stats['total']['train']}")
    print(f"Testing samples: {stats['total']['test']}")
    print("\nClass Distribution:")
    
    for class_name, counts in stats["classes"].items():
        print(f"  {class_name}:")
        print(f"    Train: {counts['train']} ({counts['train']/stats['total']['train']*100:.1f}%)")
        print(f"    Test: {counts['test']} ({counts['test']/stats['total']['test']*100:.1f}%)")
        print(f"    Total: {counts['total']}")
    
    return stats


if __name__ == "__main__":
    # Test the preprocessing module
    stats = analyze_dataset()
    print("\nCreating data loaders...")
    train_loader, test_loader, classes = create_data_loaders(batch_size=4)
    print(f"Classes: {classes}")
    print(f"Batches in train loader: {len(train_loader)}")
    print(f"Batches in test loader: {len(test_loader)}")
    
    # Test a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels: {labels}")