import torch
import torch.nn as nn
import timm

class ChickenFecesClassifier(nn.Module):
    """
    Vision Transformer (ViT) model for chicken feces classification.
    Simplified implementation with direct timm integration.
    """
    def __init__(self, num_classes=4, model_type='vit_base_patch16_224', pretrained=True):
        """
        Initialize Vision Transformer model for chicken feces classification
        
        Args:
            num_classes (int): Number of output classes
            model_type (str): Model type/architecture to use
            pretrained (bool): Whether to use pretrained weights
        """
        super(ChickenFecesClassifier, self).__init__()
        
        # Load pre-trained model
        self.model = timm.create_model(model_type, pretrained=pretrained, num_classes=num_classes)
        
        # If it's a ViT model, replace the head with custom classifier for better fine-tuning
        if 'vit' in model_type and hasattr(self.model, 'head'):
            # Get the in_features from the head
            in_features = self.model.head.in_features
            
            # Replace head with a new classifier
            self.model.head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes)
            )
    
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)

def create_model(model_type='vit_base_patch16_224', num_classes=4, pretrained=True):
    """
    Factory function to create model
    
    Args:
        model_type (str): Type of model to use
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        ChickenFecesClassifier: Model instance
    """
    return ChickenFecesClassifier(
        num_classes=num_classes,
        model_type=model_type,
        pretrained=pretrained
    )