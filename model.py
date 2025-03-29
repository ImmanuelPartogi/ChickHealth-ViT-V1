# Import library yang diperlukan
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import ViTForImageClassification, ViTFeatureExtractor
import seaborn as sns
from tqdm import tqdm
import logging
import random
import torch.nn.functional as F
import pandas as pd
from PIL import Image

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mengatur seed untuk hasil yang dapat direproduksi
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Memeriksa ketersediaan GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Menggunakan device: {device}")

# Mendefinisikan konstanta
MODEL_NAME = "google/vit-base-patch16-224"  # ViT-B/16
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5
DATA_DIR = "chicken_feces_dataset"
OUTPUT_DIR = "model_output"
CLASS_NAMES = ["Chicken_Coccidiosis", "Chicken_Healthy", "Chicken_NewCastleDisease", "Chicken_Salmonella"]
NUM_CLASSES = len(CLASS_NAMES)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kelas Dataset untuk gambar kotoran ayam
class ChickenFecesDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
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

# Fungsi untuk preprocessing dan augmentasi data
def get_transforms():
    # Feature extractor untuk ViT (menghandle preprocessing khusus yang dibutuhkan model)
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

# Fungsi untuk memuat dataset
def load_data():
    train_transform, test_transform = get_transforms()
    
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
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Ukuran dataset train: {len(train_dataset)}")
    logger.info(f"Ukuran dataset test: {len(test_dataset)}")
    
    # Mencatat distribusi kelas
    train_counts = [0] * NUM_CLASSES
    for _, label in train_dataset:
        train_counts[label] += 1
    
    test_counts = [0] * NUM_CLASSES
    for _, label in test_dataset:
        test_counts[label] += 1
    
    for i, class_name in enumerate(CLASS_NAMES):
        logger.info(f"Kelas {class_name}: {train_counts[i]} sampel train, {test_counts[i]} sampel test")
    
    return train_loader, test_loader

# Mendefinisikan model ViT-B/16 dengan fine-tuning
class ChickenFecesViT(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(ChickenFecesViT, self).__init__()
        # Memuat model pretrained ViT-B/16
        self.vit = ViTForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):
        outputs = self.vit(x)
        return outputs.logits

# Fungsi training
def train_model(model, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        progress_bar.set_postfix({
            'Loss': f"{train_loss/(batch_idx+1):.3f}",
            'Acc': f"{100.*correct/total:.2f}%"
        })
    
    return train_loss/len(train_loader), 100.*correct/total

# Fungsi validasi
def validate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Validasi"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100.*correct/total
    
    # Menghitung metrik
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    logger.info(f"Test Loss: {test_loss/len(test_loader):.3f} | Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
    
    return test_loss/len(test_loader), accuracy, all_preds, all_targets

# Fungsi untuk membuat confusion matrix
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Confusion matrix disimpan ke {save_path}")

# Loop training utama
def main():
    # Memuat data
    train_loader, test_loader = load_data()
    
    # Inisialisasi model
    model = ChickenFecesViT(num_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Mendefinisikan loss function dan optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2, 
        verbose=True
    )
    
    # Loop training
    best_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validasi
        val_loss, val_acc, all_preds, all_targets = validate_model(model, test_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Menyimpan model jika akurasi terbaik
        if val_acc > best_acc:
            logger.info(f"Model terbaik baru dengan akurasi: {val_acc:.2f}%")
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            
            # Membuat classification report
            report = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4)
            logger.info(f"Classification Report:\n{report}")
            
            # Plot confusion matrix
            plot_confusion_matrix(all_targets, all_preds, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    
    # Plot kurva training
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
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"))
    logger.info(f"Kurva training disimpan ke {os.path.join(OUTPUT_DIR, 'training_curves.png')}")
    
    # Menyimpan model akhir
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_names': CLASS_NAMES,
        'accuracy': best_acc
    }, os.path.join(OUTPUT_DIR, "final_model.pth"))
    logger.info(f"Model akhir disimpan ke {os.path.join(OUTPUT_DIR, 'final_model.pth')}")

if __name__ == "__main__":
    main()