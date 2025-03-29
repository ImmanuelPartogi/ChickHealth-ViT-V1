# Import library yang diperlukan
import os
import torch
import numpy as np
from PIL import Image
from transformers import ViTFeatureExtractor
from torchvision import transforms
from flask import Flask, render_template, request, jsonify
import io
import base64

# Custom import
from model import ChickenFecesViT, NUM_CLASSES, MODEL_NAME, IMAGE_SIZE

app = Flask(__name__, template_folder='templates')

# Memuat model terlatih
def load_model():
    model = ChickenFecesViT(num_classes=NUM_CLASSES)
    checkpoint = torch.load('model_output/final_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_names']

# Preprocessing gambar untuk prediksi
def preprocess_image(image):
    feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)
    
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    ])
    
    image = transform(image).unsqueeze(0)
    return image

# Melakukan prediksi
def predict(image, model, class_names):
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        
    prediction = {
        'class': class_names[predicted_class],
        'confidence': float(probabilities[predicted_class]) * 100,
        'probabilities': {class_names[i]: float(prob) * 100 for i, prob in enumerate(probabilities)}
    }
    
    return prediction

# Memuat model pada saat startup
model, class_names = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada bagian file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    
    try:
        # Membaca dan preprocessing gambar
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_img = preprocess_image(img)
        
        # Melakukan prediksi
        prediction = predict(processed_img, model, class_names)
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Menambahkan endpoint untuk informasi penyakit
@app.route('/disease_info', methods=['GET'])
def disease_info():
    disease_info = {
        'Chicken_Coccidiosis': {
            'description': 'Penyakit parasit yang disebabkan oleh protozoa dari genus Eimeria yang menyerang usus ayam.',
            'symptoms': 'Diare berdarah, kehilangan berat badan, penurunan produksi telur, lesu, dan peningkatan mortalitas.',
            'treatment': 'Obat anticoccidial seperti amprolium, sulfonamid, atau ionophores. Menjaga kebersihan kandang juga penting.'
        },
        'Chicken_Healthy': {
            'description': 'Ayam dalam kondisi sehat tanpa adanya penyakit.',
            'symptoms': 'Kotoran normal, aktivitas normal, nafsu makan baik, dan produksi telur normal.',
            'treatment': 'Tidak diperlukan pengobatan. Pertahankan nutrisi, kebersihan, dan manajemen yang baik.'
        },
        'Chicken_NewCastleDisease': {
            'description': 'Penyakit virus menular yang menyerang sistem pernapasan, saraf, dan pencernaan ayam.',
            'symptoms': 'Gangguan pernapasan, gejala saraf seperti kejang dan leher bengkok, diare kehijauan, dan penurunan produksi telur.',
            'treatment': 'Tidak ada pengobatan khusus. Vaksinasi rutin untuk pencegahan dan isolasi ayam yang terinfeksi.'
        },
        'Chicken_Salmonella': {
            'description': 'Infeksi bakteri dari genus Salmonella yang dapat menyebabkan penyakit pada ayam dan manusia.',
            'symptoms': 'Diare, lesu, penurunan nafsu makan, kehausan berlebih, dan peningkatan mortalitas pada anak ayam.',
            'treatment': 'Antibiotik seperti enrofloxacin atau neomycin. Peningkatan sanitasi dan biosecurity.'
        }
    }
    
    return jsonify(disease_info)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)