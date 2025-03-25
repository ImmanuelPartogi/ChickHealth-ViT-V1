import os
import torch
import numpy as np
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import logging
import time
from torchvision import transforms

# Import the model components
from model import create_model
from dataset import DEFAULT_CLASSES, IMAGE_SIZE
from predict import preprocess_image, predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('chicken_classifier_web')

app = Flask(__name__)

# Application configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['MODEL_PATH'] = os.path.join('models', 'best_vit_chicken_classifier.pth')
app.config['MODEL_TYPE'] = 'vit_base_patch16_224'  # Use Vision Transformer Base/16
app.config['CONFIDENCE_THRESHOLD'] = 0.70  # 70% confidence threshold

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    """Load the classification model"""
    global model
    try:
        if model is None:
            logger.info(f"Loading model from {app.config['MODEL_PATH']}")
            model = create_model(model_type=app.config['MODEL_TYPE'], num_classes=len(DEFAULT_CLASSES))
            model.load_state_dict(torch.load(app.config['MODEL_PATH'], map_location=device))
            model = model.to(device)
            model.eval()
            logger.info(f"Model loaded successfully on {device}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model = None
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_image(image_path):
    """
    Basic validation to ensure the image is usable
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (is_valid, message)
    """
    try:
        img = Image.open(image_path)
        
        # Check if image is too small
        width, height = img.size
        if width < 100 or height < 100:
            return False, "Image is too small (minimum 100x100 pixels)"
        
        # Convert to numpy for simple checks
        img_np = np.array(img.convert('RGB'))
        
        # Check if image is too dark or too bright
        mean_brightness = np.mean(img_np)
        if mean_brightness < 30:
            return False, "Image is too dark"
        if mean_brightness > 225:
            return False, "Image is too bright"
            
        return True, "Image is valid"
        
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return False, f"Error validating image: {str(e)}"

def get_disease_info(class_name):
    """Provides information about the disease and recommendations"""
    
    info = {
        'Chicken_Healthy': {
            'description': 'Kotoran menunjukkan ayam dalam kondisi sehat.',
            'recommendations': [
                'Pertahankan kualitas pakan dan manajemen peternakan yang baik',
                'Lanjutkan program vaksinasi secara rutin',
                'Pantau kondisi kotoran ayam secara berkala'
            ]
        },
        'Chicken_Coccidiosis': {
            'description': 'Coccidiosis adalah penyakit parasit yang disebabkan oleh protozoa Eimeria, ditandai dengan kotoran berdarah atau berwarna kemerahan.',
            'recommendations': [
                'Berikan pengobatan anticoccidial sesuai resep dokter hewan',
                'Jaga kebersihan kandang, hindari kelembaban berlebih',
                'Isolasi ayam yang terinfeksi untuk mencegah penyebaran',
                'Tingkatkan sanitasi dan desinfeksi peralatan peternakan'
            ]
        },
        'Chicken_NewCastleDisease': {
            'description': 'Newcastle Disease (ND) adalah penyakit virus yang sangat menular dengan gejala kotoran berwarna hijau atau putih berair.',
            'recommendations': [
                'Segera konsultasikan dengan dokter hewan',
                'Lakukan vaksinasi ND pada seluruh ayam di peternakan',
                'Isolasi ayam yang terinfeksi dengan ketat',
                'Lakukan desinfeksi menyeluruh pada kandang dan peralatan',
                'Pantau kondisi ayam lain untuk gejala awal'
            ]
        },
        'Chicken_Salmonella': {
            'description': 'Salmonellosis disebabkan oleh bakteri Salmonella dengan kotoran berwarna kekuningan atau putih kapur.',
            'recommendations': [
                'Berikan antibiotik sesuai resep dokter hewan',
                'Tingkatkan biosecurity di peternakan',
                'Jaga kebersihan sumber air, pakan, dan peralatan',
                'Lakukan pemeriksaan rutin untuk Salmonella pada semua ayam',
                'Isolasi ayam yang terinfeksi dan tangani secara hati-hati'
            ]
        }
    }
    
    return info.get(class_name, {
        'description': 'Informasi tidak tersedia untuk jenis kotoran ini',
        'recommendations': ['Konsultasikan dengan dokter hewan untuk diagnosa lebih lanjut']
    })

@app.route('/')
def index():
    """Main page route"""
    # Check model status
    global model
    model_error = None
    
    if model is None:
        model_loaded = load_model()
        if not model_loaded:
            model_error = f"Model tidak ditemukan di {app.config['MODEL_PATH']}. Pastikan model telah dilatih dan disimpan."
    
    return render_template('index.html', model_error=model_error)

@app.route('/predict', methods=['POST'])
def predict_image():
    """Prediction endpoint"""
    start_time = time.time()
    
    # Check if file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    
    # Check if file has a name
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check file format
    if file and allowed_file(file.filename):
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Validate image
        is_valid, validation_message = validate_image(file_path)
        if not is_valid:
            return jsonify({
                'error': f"Image validation failed: {validation_message}",
                'image_validation_failed': True
            }), 400
        
        # Check if model is loaded
        global model
        if model is None:
            model_loaded = load_model()
            if not model_loaded:
                return jsonify({'error': 'Model not found or failed to load'}), 500
        
        try:
            # Preprocess image
            image_tensor = preprocess_image(file_path)
            
            # Predict
            class_name, confidence, all_scores = predict(model, image_tensor, device, DEFAULT_CLASSES)
            
            # Check confidence threshold
            if confidence < app.config['CONFIDENCE_THRESHOLD'] * 100:
                return jsonify({
                    'is_chicken_feces': False,
                    'message': 'This image might not be chicken feces or is difficult to classify',
                    'confidence': float(confidence),
                    'scores': [{
                        'class': cls,
                        'score': float(score)
                    } for cls, score in all_scores],
                    'prediction': class_name,
                    'low_confidence': True
                })
            
            # Get disease information
            info = get_disease_info(class_name)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Format results
            results = {
                'prediction': class_name,
                'confidence': float(confidence),
                'is_chicken_feces': True,
                'scores': [{
                    'class': cls,
                    'score': float(score)
                } for cls, score in all_scores],
                'image_path': f"{file_path}",
                'info': info,
                'processing_time': processing_time
            }
            
            return jsonify(results)
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': f'Error during prediction: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file format. Please upload a JPG, JPEG, or PNG image.'}), 400

@app.route('/model_status', methods=['GET'])
def get_model_status():
    """Check model status and provide information"""
    global model
    
    if model is None:
        model_loaded = load_model()
        status = "loaded" if model_loaded else "not_loaded"
    else:
        status = "loaded"
    
    return jsonify({
        'status': status,
        'model_type': app.config['MODEL_TYPE'],
        'device': str(device)
    })

if __name__ == '__main__':
    # Load model at startup
    load_model()
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)