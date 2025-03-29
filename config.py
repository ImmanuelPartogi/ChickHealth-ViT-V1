"""
Configuration parameters for the chicken disease classification project.
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "chicken_feces_dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "model_output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Model configuration
MODEL_NAME = "google/vit-base-patch16-224"  # ViT-B/16
CLASS_NAMES = ["Chicken_Coccidiosis", "Chicken_Healthy", "Chicken_NewCastleDisease", "Chicken_Salmonella"]
NUM_CLASSES = len(CLASS_NAMES)

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
IMAGE_SIZE = 224
RANDOM_SEED = 42

# Evaluation parameters
BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "final_model.pth")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
TRAINING_CURVES_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")

# Application parameters
APP_HOST = "0.0.0.0"
APP_PORT = 5000
DEBUG_MODE = True

# Disease information
DISEASE_INFO = {
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