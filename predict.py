import os
import argparse
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from model import create_model
from dataset import DEFAULT_CLASSES, IMAGE_SIZE

def preprocess_image(image_path, img_size=IMAGE_SIZE):
    """
    Preprocess an image for model inference
    
    Args:
        image_path (str): Path to input image
        img_size (int): Size to resize image to
    
    Returns:
        tensor: Preprocessed image tensor
    """
    # Transformations for inference
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Open and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def predict(model, image_tensor, device='cuda', classes=DEFAULT_CLASSES):
    """
    Make prediction on an image
    
    Args:
        model: PyTorch model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        classes: List of class names
    
    Returns:
        tuple: (predicted_class, confidence, all_scores)
    """
    # Move tensor to appropriate device
    image_tensor = image_tensor.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # Get class name and confidence
    class_name = classes[predicted_class]
    confidence = probabilities[predicted_class].item() * 100
    
    # Get all confidence scores
    all_scores = [(classes[i], prob.item() * 100) for i, prob in enumerate(probabilities)]
    all_scores.sort(key=lambda x: x[1], reverse=True)
    
    return class_name, confidence, all_scores

def visualize_prediction(image_path, class_name, confidence, all_scores, output_path=None):
    """
    Visualize the prediction with the original image
    
    Args:
        image_path: Path to original image
        class_name: Predicted class name
        confidence: Confidence score for prediction
        all_scores: All confidence scores for all classes
        output_path: Path to save visualization (optional)
    """
    # Open the original image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title(f"Prediction: {class_name.replace('Chicken_', '')}\nConfidence: {confidence:.2f}%", 
                 fontsize=14)
    ax1.axis('off')
    
    # Display confidence scores for all classes
    class_names = [x[0].replace('Chicken_', '') for x in all_scores]  # Remove "Chicken_" prefix for clarity
    scores = [x[1] for x in all_scores]
    
    # Create horizontal bar chart
    bars = ax2.barh(class_names, scores, color='skyblue')
    
    # Highlight the predicted class
    for i, (cls, score) in enumerate(all_scores):
        if cls == class_name:
            bars[i].set_color('navy')
    
    # Add percentage labels to bars
    for i, v in enumerate(scores):
        ax2.text(v + 1, i, f"{v:.1f}%", va='center')
    
    ax2.set_xlabel('Confidence (%)')
    ax2.set_title('Class Probabilities')
    ax2.set_xlim(0, 105)  # Add some room for the text
    
    plt.tight_layout()
    
    # Save or display
    if output_path:
        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description='Predict chicken health condition from fecal image')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='models/best_vit_chicken_classifier.pth',
                        help='Path to the trained model')
    parser.add_argument('--model_type', type=str, default='vit_base_patch16_224',
                        help='Type of model')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize the prediction')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save visualization')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA even if available')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Check if image exists
        if not os.path.isfile(args.image_path):
            print(f"Error: Image {args.image_path} not found")
            return
        
        # Create model
        model = create_model(model_type=args.model_type, num_classes=len(DEFAULT_CLASSES))
        
        # Load model weights
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # Preprocess image
        image_tensor = preprocess_image(args.image_path)
        
        # Make prediction
        class_name, confidence, all_scores = predict(model, image_tensor, device, DEFAULT_CLASSES)
        
        # Print results
        print(f"\nPrediction: {class_name}")
        print(f"Confidence: {confidence:.2f}%\n")
        print("All class probabilities:")
        for cls, score in all_scores:
            print(f"  {cls}: {score:.2f}%")
        
        # Visualize if requested
        if args.visualize:
            visualize_prediction(args.image_path, class_name, confidence, all_scores, args.output_path)
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()