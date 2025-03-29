# Klasifikasi Penyakit Ayam dari Kotoran dengan Vision Transformer (ViT-B/16)

Proyek ini merupakan implementasi deep learning untuk mengklasifikasikan penyakit ayam berdasarkan gambar kotoran menggunakan arsitektur Vision Transformer (ViT-B/16).

## Tentang Proyek

Klasifikasi ini mampu mendeteksi 4 kondisi berbeda pada ayam:
- Coccidiosis
- Healthy (Sehat)
- Newcastle Disease
- Salmonella

Model menggunakan transfer learning dari pre-trained ViT-B/16 dan fine-tuning untuk mencapai akurasi tinggi dalam klasifikasi.

## Struktur Proyek

```
chicken_disease_classification/
│
├── model.py                 # Implementasi model ViT-B/16 dan kode training
├── app.py                   # Aplikasi Flask untuk integrasi web
├── requirements.txt         # Daftar library yang diperlukan
│
├── templates/               
│   └── index.html           # Halaman utama aplikasi web
│
├── model_output/            
│   ├── best_model.pth       # Model dengan akurasi terbaik
│   ├── final_model.pth      # Model final
│   ├── confusion_matrix.png # Visualisasi confusion matrix
│   └── training_curves.png  # Grafik loss dan akurasi selama training
│
├── chicken_feces_dataset/   
│   ├── train/               
│   │   ├── Chicken_Coccidiosis/
│   │   ├── Chicken_Healthy/
│   │   ├── Chicken_NewCastleDisease/
│   │   └── Chicken_Salmonella/
│   │
│   └── test/                
│       ├── Chicken_Coccidiosis/
│       ├── Chicken_Healthy/
│       ├── Chicken_NewCastleDisease/
│       └── Chicken_Salmonella/
│
└── README.md                # Dokumentasi project
```

## Requirement

- Python 3.8+
- PyTorch 2.1.0+
- Transformers 4.35.0+
- Flask 2.3.3+
- Dan library lain yang tercantum di `requirements.txt`

## Cara Penggunaan

### 1. Setup Environment

```bash
# Clone repositori
git clone https://github.com/username/chicken-disease-classification.git
cd chicken-disease-classification

# Buat dan aktifkan virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Persiapan Dataset

Letakkan dataset gambar kotoran ayam dengan struktur berikut:
```
chicken_feces_dataset/
├── train/
│   ├── Chicken_Coccidiosis/
│   ├── Chicken_Healthy/
│   ├── Chicken_NewCastleDisease/
│   └── Chicken_Salmonella/
└── test/
    ├── Chicken_Coccidiosis/
    ├── Chicken_Healthy/
    ├── Chicken_NewCastleDisease/
    └── Chicken_Salmonella/
```

### 3. Training Model

```bash
python model.py
```

Proses training akan memakan waktu tergantung pada hardware yang tersedia. Hasil training berupa model dan metrik evaluasi akan disimpan di direktori `model_output/`.

### 4. Menjalankan Aplikasi Web

```bash
python app.py
```

Kemudian buka browser dan akses `http://localhost:5000` untuk menggunakan aplikasi klasifikasi.

## Fitur Aplikasi Web

- Upload gambar kotoran ayam
- Hasil klasifikasi dengan persentase kepercayaan
- Visualisasi probabilitas untuk setiap kelas
- Informasi mengenai penyakit yang terdeteksi

## Performa Model

Model yang dilatih menggunakan arsitektur ViT-B/16 dengan teknik transfer learning mampu mencapai:
- Akurasi: ~95% (tergantung pada dataset)
- Precision: ~94%
- Recall: ~93%
- F1-Score: ~93%

## Penulis

[Nama Anda]

## Lisensi

MIT License