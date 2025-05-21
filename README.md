# SI PENDEKAR (Sistem Deteksi dan Prediksi Kerusakan Jalan Berbasis AI)

**Lomba Teknologi Tepat Guna (TTG) Kabupaten Indramayu**

---

## 📝 Deskripsi Singkat

SI PENDEKAR adalah sistem berbasis AI yang mampu:

* Mengidentifikasi jenis kerusakan jalan dari gambar.
* Mengukur ukuran kerusakan (bounding box dan dimensi).
* Memperkirakan waktu perbaikan secara otomatis.
* Merekomendasikan jenis bahan dan jumlah bahan yang diperlukan untuk perbaikan.

Dikembangkan dengan pendekatan **multitask transfer learning** menggunakan **Swin Transformer** untuk menghasilkan prediksi lengkap hanya dari satu input gambar.

---

## 🚀 Fitur Utama

1. **Klasifikasi Kerusakan**: Deteksi tipe kerusakan (retak, lubang, dll.)
2. **Estimasi Ukuran**: Output koordinat `(xmin, ymin, xmax, ymax)` beserta ukuran fisik kerusakan.
3. **Prediksi Waktu Perbaikan**: Waktu (dalam jam) yang dibutuhkan untuk perbaikan.
4. **Rekomendasi Material**: Jenis bahan perbaikan (aspal, semen, dll.) dan estimasi jumlah (Kg).
5. **Integrasi Full Stack**: Backend Flask API & Laravel, siap dikonsumsi aplikasi web.

---

## 📦 Struktur Dataset

Dataset asli diperoleh dari Google Drive dosen (folder ID: `1rzlWb0y0Cw9Syy-wOHkjHjCe7WEUO--Y`) dengan struktur:

```
pavementscapes/
├── img/
│   ├── train/ (2.500 images)
│   ├── test/  (1.000 images)
│   └── val/   (1.000 images)
├── mask/
│   ├── train/ (2.500 images)
│   ├── test/  (1.000 images)
│   └── val/   (1.000 images)
├── bbox_label_train.csv
├── bbox_label_test.csv
└── bbox_label_val.csv
```

**Peran Dataset Enrichment:**

* Mengembangkan kolom di CSV dari `(xmin, ymin, xmax, ymax, jenis_kerusakan)`
  menjadi `(xmin, ymin, xmax, ymax, jenis_kerusakan, ukuran, estimasi_waktu, jenis_bahan, jumlah_bahan)`.

---

## 🔧 Arsitektur & Teknologi

* **Model**  : MultiTaskSwin (Swin Transformer multitask) dengan lima output heads.
* **Library**: PyTorch, TIMM, Albumentations, scikit-learn (LabelEncoder, scaler).
* **Backend**: Flask API (`/predict` route) dengan TTA (Test Time Augmentation).
* **Frontend**: Laravel (ditangani tim FE/API).
* **Deployment**: Docker-ready; dapat diintegrasikan ke aplikasi web atau mobile.

---

## 🛠️ Output Model

| File                  | Keterangan                          |
| --------------------- | ----------------------------------- |
| `best_model.pth`      | Bobot model multitask Swin          |
| `damage_le.pkl`       | Label encoder untuk kelas kerusakan |
| `material_le.pkl`     | Label encoder untuk jenis bahan     |
| `time_scaler.pkl`     | Scaler untuk normalisasi waktu      |
| `quantity_scaler.pkl` | Scaler untuk normalisasi volume     |

---

## 👥 Tim & Peran

* **Ketua & Pembimbing** : Putri Rizqiyah (pengaju & pendamping proyek)
* **FE & API Developer** : Sean Andrianto (Laravel & integrasi Flask API)
* **Machine Learning Engineer** : *Ferri Krisdiantoro* (dataset enrichment & training model)

---

## ⚙️ Cara Menjalankan

1. Clone repository:

   ```bash
   git clone <repo-url>
   cd <project-folder>
   ```
2. Download dataset & model:

   ```bash
   python download.py
   ```
3. Jalankan API Flask:

   ```bash
   cd flask_app
   python app.py
   ```
4. Akses endpoint:

   ```http
   POST /predict  (form-data: key `image`)
   ```

---

## 📈 Hasil & Manfaat

* **Akurasi**: Tinggi untuk deteksi dan prediksi multitask.
* **Efisiensi**: Mempercepat proses identifikasi dan estimasi kebutuhan material.
* **Dampak Sosial**: Mendukung perbaikan infrastruktur jalan yang lebih tepat sasaran.

---

## 📞 Kontak

Untuk demo atau kolaborasi, hubungi: *[ferryk935@gmail.com](mailto:ferryk935@gmail.com)* atau kunjungi profil LinkedIn: [linkedin.com/in/ferrikrisdiantoro](https://www.linkedin.com/in/ferrikrisdiantoro)
