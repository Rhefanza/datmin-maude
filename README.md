# datmin-maude

Klasifikasi narasi laporan insiden alat MRI (FDA MAUDE/openFDA, kode produk LNH) ke 4 kelas:
Injury, Malfunction, Death, Other.

## Urutan notebook

| Notebook | Isi | Masukan → keluaran |
|---|---|---|
| `notebooks/01_cleaning_eda.ipynb` | cleaning (narasi ≥10 kata, buang duplikat persis dan mirip, grup kemiripan) dan EDA | `data/raw/maude_mri_LNH.csv` → `data/interim/data_bersih.csv` (3.604 laporan) |
| `notebooks/02_prapemrosesan_split.ipynb` | buang kata pembocor label, dua jalur teks, 5-fold per grup, split temporal | `data_bersih.csv` → `data/processed/data_eksperimen.csv` |
| `notebooks/03_Modeling,_Evaluation,_and_Error_Analysis.ipynb` | modeling 5-fold OOF (TF-IDF, BiomedBERT, DeBERTa zero-shot), McNemar, telaah kesalahan (Colab) | `data_eksperimen.csv` → `outputs/predictions/`, `outputs/tables03/` |
| `notebooks/08_eksperimen_lengkap.ipynb` | eksperimen lengkap train/val/test (70/15/15 per grup) di Kaggle GPU T4 | `maude_mri_LNH.csv` → `outputs/nb08/` |

Notebook 01 dan 02 bisa dijalankan dari clone repo ini (lokal/Colab) atau di Kaggle.
Notebook 08 dijalankan di Kaggle (GPU, ±2 jam). Keluarannya ada di `outputs/nb08/`, tetapi file notebook di repo tidak berisi keluaran sel.

Dasar paper untuk setiap metode di notebook 08 (DOI sudah diperiksa, ditandai sama/modifikasi/data proyek) ada di [`docs/dasar_paper_metode.md`](docs/dasar_paper_metode.md).

## Ringkasan hasil notebook 08 (test, 534 laporan)

Rancangan eksperimen:
- 6 skenario per model: {tanpa, bobot kelas, SMOTE untuk ML / ROS untuk neural} × {tanpa augmentasi, back-translation en→de→en}.
- Augmentasi back-translation hanya untuk Death dan Other, dan hanya pada data train.
- Konfigurasi terbaik tiap keluarga dipilih menurut macro-F1 val.

| Keluarga | Model terbaik (menurut val) | macro-F1 test | akurasi | F1 Death |
|---|---|---|---|---|
| ML | TF-IDF kata + SVM linear, SMOTE | 0,606 | 0,839 | 0,286 |
| DL | BiLSTM, ROS + aug | 0,583 | 0,824 | 0,182 |
| DL hybrid | BiomedBERT + BiLSTM + attention, bobot + aug | 0,669 | 0,843 | 0,533 |
| DL hybrid + HP | sama, 8 trial random search | 0,662 | 0,852 | 0,500 |
| **BART-large-MNLI fine-tune** | ROS + aug | **0,722** | **0,856** | **0,727** |

Catatan pembacaan:
- **Uji McNemar.** Setelah koreksi Holm, BART tidak berbeda bermakna dari model terbaik keluarga lain (semua p_holm ≥ 0,18).
- **Kelas Death.** Test hanya memuat 6 laporan Death, sehingga satu laporan mengubah recall Death sebesar 17 poin persen.
- **Kelas tersulit: Other.** F1 Other hanya 0,40. Kesalahan terbanyak muncul karena kosakata yang tumpang tindih antarkelas dan gaya laporan Other lama. Rinciannya ada di `outputs/nb08/tables/15_*.csv` dan `figures/15_alasan_kesalahan.png`.
- **Selisih val dan test.** Macro-F1 val (±0,8) lebih tinggi daripada test (±0,6–0,7). Penyebabnya, val hanya memuat 4 laporan Death, dan val juga dipakai untuk memilih konfigurasi.
