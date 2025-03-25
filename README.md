# Chicken Feces Disease Classification System

This project uses deep learning (Vision Transformer - ViT-B/16) to analyze and classify chicken feces images to detect diseases.

## Overview

The system classifies chicken feces images into four categories:
- Chicken_Coccidiosis
- Chicken_Healthy
- Chicken_NewCastleDisease
- Chicken_Salmonella

## Project Structure

```
.
├── app.py              # Flask web application
├── dataset.py          # Data loading and processing
├── model.py            # Model architecture
├── predict.py          # Prediction functionality
├── train.py            # Training script
├── uploads/            # Uploaded images for prediction
├── models/             # Saved model weights
└── templates/          # HTML templates for web interface
    └── index.html      # Main page template
```

## Requirements

The project requires the following Python packages:
- torch
- torchvision
- timm
- numpy
- flask
- pillow
- scikit-learn
- matplotlib
- seaborn
- tqdm

You can install them using the included requirements.txt file:

```bash
pip install -r requirements.txt
```

## Training the Model

To train the model, use the train.py script:

```bash
python train.py --data_dir path/to/dataset --output_dir ./output --num_epochs 20 --batch_size 16
```

Important arguments:
- `--data_dir`: Path to the dataset directory (required)
- `--output_dir`: Directory to save model weights and results (default: .)
- `--num_epochs`: Number of epochs for training (default: 20)
- `--batch_size`: Batch size for training (default: 16)
- `--learning_rate`: Initial learning rate (default: 0.0005)
- `--model_type`: Model type to use (default: vit_base_patch16_224)
- `--no_cuda`: Disable CUDA even if available

The dataset directory should have the following structure:
```
dataset/
├── train/                  # Training images
│   ├── Chicken_Coccidiosis/
│   ├── Chicken_Healthy/
│   ├── Chicken_NewCastleDisease/
│   └── Chicken_Salmonella/
└── test/                   # Test/validation images
    ├── Chicken_Coccidiosis/
    ├── Chicken_Healthy/
    ├── Chicken_NewCastleDisease/
    └── Chicken_Salmonella/
```

Alternatively, if you don't have a train/test split, you can use a flat structure and the code will automatically create a split:
```
dataset/
├── Chicken_Coccidiosis/
├── Chicken_Healthy/
├── Chicken_NewCastleDisease/
└── Chicken_Salmonella/
```

## Making Predictions

You can use the predict.py script to make predictions on individual images:

```bash
python predict.py --image_path path/to/image.jpg --model_path models/best_vit_chicken_classifier.pth --visualize
```

Arguments:
- `--image_path`: Path to the input image (required)
- `--model_path`: Path to the trained model (default: models/best_vit_chicken_classifier.pth)
- `--visualize`: Flag to visualize the prediction
- `--output_path`: Path to save visualization (optional)
- `--no_cuda`: Disable CUDA even if available

## Web Application

To start the web application:

```bash
python app.py
```

This starts a Flask web server at http://localhost:5000/ where you can upload images for classification.

## Model Architecture

The system uses a Vision Transformer (ViT-B/16) model, which has been simplified while maintaining high accuracy. The model consists of:

1. A pre-trained ViT-B/16 backbone
2. A custom classifier head optimized for the chicken feces classification task

## Improvements from Previous Version

This implementation has been significantly simplified compared to the previous version:

1. **Reduced Complexity**: Removed unnecessary complexity in all components
2. **Streamlined Training**: Simplified training process with effective but lighter augmentation
3. **Optimized Model**: Streamlined model architecture while maintaining accuracy
4. **Memory Efficiency**: Reduced memory footprint during training
5. **Simplified Web App**: Kept core functionality while removing unnecessary complexity

## Performance Expectations

On a modern GPU, training should complete in a few hours even with a substantial dataset. The simplified model is expected to achieve accuracy comparable to the original, more complex implementation.

## Notes for CPU Training

If training on a CPU:
- Batch size is automatically reduced to avoid memory issues
- Number of workers is reduced
- Some advanced features are disabled

These adjustments ensure the model can be trained on less powerful hardware, albeit more slowly.