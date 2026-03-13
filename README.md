# IF3270 Pembelajaran Mesin - Tugas Besar 1
**Implementasi Feedforward Neural Network (FFNN) *From Scratch***

## 📖 Deskripsi Singkat
Repositori ini berisi implementasi model *Feedforward Neural Network* (FFNN) yang dibangun sepenuhnya dari awal (*from scratch*) menggunakan bahasa pemrograman Python dan pustaka komputasi numerik NumPy. Pembangunan model ini tidak menggunakan *library machine learning* tingkat tinggi (seperti scikit-learn, TensorFlow, atau PyTorch) untuk algoritma intinya. Proyek ini dikembangkan untuk memenuhi Tugas Besar 1 mata kuliah IF3270 Pembelajaran Mesin.

Model FFNN yang diimplementasikan pada repositori ini mendukung berbagai fitur, antara lain:
- Kustomisasi dinamis untuk jumlah *hidden layer* dan *neuron* per lapisan.
- Pilihan fungsi aktivasi: `Linear`, `ReLU`, `Sigmoid`, `Tanh`, dan `Softmax`.
- Pilihan fungsi kerugian (*Loss Functions*): `MSE`, `BCE`, dan `CCE`.
- Algoritma pembelajaran menggunakan *Backpropagation* (aturan rantai / *chain rule*) dengan optimasi *Gradient Descent*.
- Mekanisme regularisasi bobot (L1 dan L2 / *Weight Decay*).
- Berbagai metode inisialisasi bobot (`Zero`, `Random Uniform`, dan `Random Normal`).

Sebagai kasus uji, model ini diimplementasikan untuk menyelesaikan persoalan klasifikasi menggunakan dataset demografi dan akademik **Global Student Placement & Salary**.

## 📂 Struktur Direktori Utama
```text
ML-IzinkanKamiMelawanAmerika/
│
├── src/
│   ├── activations.py    # Implementasi kelas berbagai fungsi aktivasi
│   ├── ffnn.py           # Arsitektur utama kelas Feedforward Neural Network
│   ├── layer.py          # Implementasi kelas DenseLayer (operasi matriks per lapisan)
│   ├── losses.py         # Implementasi kelas fungsi objektif kerugian
│   ├── utils.py          # Utilitas untuk visualisasi metrik dan perhitungan statistik
│   │
│   ├── data/
│   │   └── student_placement_salary.csv  # Dataset mentah
│   │
│   └── notebooks/
│       └── experiment.ipynb              # File utama eksekusi pelatihan dan pengujian model
│
├── doc/
│   └── Laporan Tugas Besar 1 Machine Learning.pdf # Laporan analisis komprehensif
│
└── README.md
```

## ⚙️ Setup dan Instalasi
Pastikan sistem Anda sudah terinstal **Python 3.8** atau versi yang lebih baru. Ikuti langkah-langkah berikut untuk mengatur *environment* lokal Anda:

1. **Clone repositori ini ke komputer lokal:**
   ```bash
   git clone https://github.com/fithrarzk/ML-IzinkanKamiMelawanAmerika.git
   cd ML-IzinkanKamiMelawanAmerika
   ```

2. **Instal seluruh dependensi yang dibutuhkan:**
   Library utama yang dibutuhkan adalah `numpy` (untuk komputasi inti), `pandas`, `matplotlib` (untuk visualisasi), `scikit-learn` (hanya digunakan untuk pra-pemrosesan data dan *benchmarking* model), serta `jupyter` untuk menjalankan *notebook*.
   ```bash
   pip install numpy pandas matplotlib scikit-learn jupyter
   ```

## 🚀 Cara Menjalankan Program
Seluruh eksperimen yang mencakup pra-pemrosesan data, pelatihan model (*training*), penyetelan *hyperparameter*, visualisasi metrik, hingga *benchmarking* dieksekusi secara interaktif melalui Jupyter Notebook.

1. Buka terminal/CMD dan pastikan Anda berada di direktori utama (root) repositori.
2. Jalankan server Jupyter Notebook dengan perintah:
   ```bash
   jupyter notebook
   ```
3. Pada antarmuka peramban (*browser*) yang terbuka, navigasikan ke direktori `src/notebooks/` dan buka file `experiment.ipynb`.
4. Untuk melihat seluruh hasil dari awal hingga akhir, klik menu **Kernel** -> **Restart & Run All** (atau jalankan setiap *cell* secara berurutan dari atas ke bawah).

## 👥 Pembagian Tugas

Proyek ini dikembangkan oleh kelompok dengan rincian kontribusi sebagai berikut:

| Nama Lengkap | NIM | Peran & Deskripsi Tugas | Kontribusi |
| :--- | :--- | :--- | :---: |
| **Raka Daffa Iftikhaar** | `13523018` | isi | **33%** |
| **Muhammad Fithra Rizki** | `13523049` | isi | **33%** |
| **Muhammad Timur Kanigara** | `13523055` | isi | **34%** |