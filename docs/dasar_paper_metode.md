# Dasar paper untuk setiap metode (notebook 08)

Disusun 23 Nov 2026. Notebook acuan: `notebooks/08_eksperimen_lengkap.ipynb`. Notebook 03 tidak dipakai sebagai acuan peta ini.

Aturan proyek: rujukan terbit tahun **2021 ke atas**. Satu-satunya pengecualian adalah **paper asal model atau metode** yang memang terbit lebih awal (BART 2020, metode entailment 2019). Keduanya ditandai ⚠ dan selalu didampingi paper ≥2021 yang memakainya.

**Arti kolom "Hubungan":**
- **Sama**: kita memakai metode seperti di paper.
- **Modifikasi**: kita mengubah sesuatu dari paper; perubahannya disebut.
- **Data proyek**: pilihan ini didasarkan pada angka dari data kita sendiri, karena belum ada paper yang menetapkan angkanya.

**Arti kolom "Cek":**
- **T**: isi yang dikutip sudah dicek langsung di teks penuh.
- **A**: sudah dicek di abstrak.
- **M**: baru metadatanya (DOI, judul, tahun) yang dicek lewat Crossref; isinya belum.

Semua DOI di bawah sudah dicek lewat Crossref.

## 1. Urgensi dan celah penelitian

| Klaim di laporan | Paper | Cek |
|---|---|---|
| Kode `event_type` MAUDE tidak andal. Sebagian laporan yang memuat kematian tidak dikode Death (perkiraan 23%). | Lalani dkk. 2021, *JAMA Intern Med*, [10.1001/jamainternmed.2021.3942](https://doi.org/10.1001/jamainternmed.2021.3942) | A |
| Narasi laporan efek samping alat (dari MAUDE) bisa diklasifikasikan dengan deep learning (BERT-BiLSTM-attention, 4.568 laporan da Vinci, klasifikasi biner bahaya/tidak). | Li dkk. 2024, *Front Comput Neurosci*, [10.3389/fncom.2024.1476164](https://doi.org/10.3389/fncom.2024.1476164) | T |
| LLM dipakai untuk mengode ulang Event Type MAUDE, disertai telaah manusia. | Li dkk. 2026, *npj Digit Med*, [10.1038/s41746-026-02353-7](https://doi.org/10.1038/s41746-026-02353-7) | T (diskusi Codex) |
| Studi MRI MAUDE 2018–2024 masih memakai kata kunci dan telaah pakar, bukan prediksi 4 kode. **Ini celah yang kita isi.** | Lee dkk. 2026, *J Anal Sci Technol*, [10.1186/s40543-026-00559-8](https://doi.org/10.1186/s40543-026-00559-8) | T (diskusi Codex) |
| TF-IDF + SVM sudah dipakai untuk menyaring laporan MAUDE (kejadian alat berbasis ML). | Wang dkk. 2024, *Stud Health Technol Inform* (MEDINFO 2023), [10.3233/SHTI231036](https://doi.org/10.3233/SHTI231036) | A |
| Ada benchmark klasifikasi multi-label MAUDE dengan evaluasi temporal. | Agarwal dkk. 2026, *ACL*, [10.18653/v1/2026.acl-long.2148](https://doi.org/10.18653/v1/2026.acl-long.2148) | M |

## 2. Cleaning

| Metode kita | Paper dasar | Hubungan | Cek |
|---|---|---|---|
| Hanya `event_text` yang menjadi fitur. `patient_outcome` dan `mfr_narrative` dibuang karena membocorkan label. | Kapoor & Narayanan 2023, *Patterns*, [10.1016/j.patter.2023.100804](https://doi.org/10.1016/j.patter.2023.100804): fitur yang tidak tersedia saat prediksi adalah salah satu jenis kebocoran. | Sama | A |
| Narasi minimal 10 kata | Tidak ada paper yang menetapkan angka 10. | Data proyek: yang gugur hanya 1 Death, berisi 1 kata, dan tidak menyebut kematian. | — |
| Buang duplikat persis (teks dinormalisasi) | Lee dkk. 2022, *ACL*, [10.18653/v1/2022.acl-long.577](https://doi.org/10.18653/v1/2022.acl-long.577): duplikat persis dan duplikat mirip membuat evaluasi bias. | Sama | A |
| Buang duplikat mirip (cosine TF-IDF ≥ 0,9, dalam kelas yang sama) | Lee dkk. 2022: mereka mendeteksi duplikat mirip dengan MinHash. | Modifikasi: kita memakai cosine TF-IDF, karena data kita kecil (4.645 laporan) sehingga semua pasangan bisa dihitung langsung. Ambang 0,9 ditetapkan dari data proyek. | A |
| Grup kemiripan (komponen terhubung) dijaga agar tidak terpisah ke split yang berbeda | Kapoor & Narayanan 2023: data latih dan uji yang tidak independen adalah sumber kebocoran. | Sama | A |

## 3. Praproses

| Metode kita | Paper dasar | Hubungan | Cek |
|---|---|---|---|
| Hapus kata pembocor (`injur*`, `serious*`, `harm*`, `malfunction*`, `death(s)`) | Kapoor & Narayanan 2023: input yang secara langsung memuat nama label adalah kebocoran. | Modifikasi: daftar kata ditentukan dari data proyek. Frasa "serious injury" muncul di 48,5% laporan Malfunction, tetapi hanya di 5,4% laporan Injury. | A |
| Dua jalur teks: `text_clean` untuk TF-IDF dan `text_tf` (tanda baca dipertahankan) untuk transformer | Gu dkk. 2022: model biomedis memakai tokenizer subkata sendiri, jadi teks tidak perlu di-stemming. | Modifikasi | A |
| Split 70/15/15 per grup dengan StratifiedGroupKFold | Kapoor & Narayanan 2023 | Sama | A |

## 4. Penanganan ketidakseimbangan dan augmentasi (6 skenario)

| Metode kita | Paper dasar | Hubungan | Cek |
|---|---|---|---|
| Ringkasan tiga keluarga penanganan ketidakseimbangan (resampling, augmentasi, bobot loss) | Henning dkk. 2023, *EACL*, [10.18653/v1/2023.eacl-main.38](https://doi.org/10.18653/v1/2023.eacl-main.38). Ketiganya disebut sebagai pilihan sederhana yang layak dicoba lebih dulu. | Sama: perbandingan tanpa / bobot / resample | A |
| SMOTE pada fitur TF-IDF untuk model ML | Mujahid dkk. 2024, *J Big Data*, [10.1186/s40537-024-00943-4](https://doi.org/10.1186/s40537-024-00943-4): SMOTE dan variannya diterapkan pada fitur TF-IDF/BoW dengan RF, SVM, LR, dan lainnya. | Sama | T |
| ROS (sampler seimbang) untuk model neural, pengganti SMOTE | Henning dkk. 2023 (resampling untuk deep learning) | Modifikasi: SMOTE menginterpolasi vektor fitur, sehingga tidak bermakna untuk urutan token. | A |
| Bobot kelas pada loss | Henning dkk. 2023 (loss-based) | Sama | A |
| Back-translation en→de→en (MarianMT) | Shorten dkk. 2021, *J Big Data*, [10.1186/s40537-021-00492-0](https://doi.org/10.1186/s40537-021-00492-0), menyebut back-translation sebagai augmentasi neural. Bayer dkk. 2023, *ACM Comput Surv*, [10.1145/3544558](https://doi.org/10.1145/3544558), adalah survei augmentasi untuk klasifikasi teks. | Modifikasi: hanya untuk kelas minor (Death, Other), hanya pada data train, dan kata pembocor dihapus ulang setelah diterjemahkan. | T (Shorten), M (Bayer) |

## 5. Model

| Keluarga | Metode kita | Paper dasar | Hubungan | Cek |
|---|---|---|---|---|
| ML | TF-IDF + NB / LR / SVM / RF, dan TF-IDF karakter + SVM | Wang dkk. 2024 (TF-IDF + SVM pada MAUDE). Mujahid dkk. 2024 (TF-IDF + RF/SVM/LR). | Sama. Varian TF-IDF karakter adalah modifikasi untuk menangani salah ketik dan istilah teknis. | A / T |
| DL | BiLSTM dari nol | Li dkk. 2024 memakai BiLSTM, LSTM, dan GRU sebagai baseline pada laporan efek samping MAUDE. | Sama (sebagai baseline) | A |
| DL hybrid | BiomedBERT + BiLSTM + attention, ditambah [CLS] | Li dkk. 2024: BERT-BiLSTM-Att_dropout. Gu dkk. 2022, *ACM Trans Comput Healthc*, [10.1145/3458754](https://doi.org/10.1145/3458754), tentang PubMedBERT/BiomedBERT. Prabhakar & Won 2021, *Comput Intell Neurosci*, [10.1155/2021/9425655](https://doi.org/10.1155/2021/9425655): RNN dua arah + attention untuk teks medis. | Modifikasi: BERT umum diganti BiomedBERT (Gu 2022), vektor [CLS] digabungkan, dan tugasnya 4 kelas, bukan biner. | T (Li), A (Gu, Prabhakar) |
| DL hybrid + HP | Random search 8 trial (lr encoder, lr kepala, dropout, unit BiLSTM, batch), dipilih dengan data val | Bischl dkk. 2023, *WIREs DMKD*, [10.1002/widm.1484](https://doi.org/10.1002/widm.1484) | Sama | A |
| BART fine-tune | BART-large-MNLI di-fine-tune pada semua data train, kepala MNLI diganti 4 kelas | ⚠ Lewis dkk. 2020, *ACL*, [10.18653/v1/2020.acl-main.703](https://doi.org/10.18653/v1/2020.acl-main.703) (paper asal model). ⚠ Yin dkk. 2019, *EMNLP*, [10.18653/v1/D19-1404](https://doi.org/10.18653/v1/D19-1404) (paper asal pendekatan entailment). Percha dkk. 2021, *JAMIA*, [10.1093/jamia/ocab243](https://doi.org/10.1093/jamia/ocab243): BART dipakai sebagai model NLI pada teks klinis. | Modifikasi: model tidak dipakai zero-shot. Bobot MNLI hanya dipakai sebagai titik awal, lalu model di-fine-tune untuk 4 kelas. | A (Percha), M (Lewis, Yin) |

## 6. Evaluasi dan analisis kesalahan

| Metode kita | Paper dasar | Hubungan | Cek |
|---|---|---|---|
| Macro-F1 sebagai metrik utama, ditambah F1 per kelas | Rainio dkk. 2024, *Sci Rep*, [10.1038/s41598-024-56706-x](https://doi.org/10.1038/s41598-024-56706-x), menjelaskan macro-averaging untuk klasifikasi multi-kelas. | Sama | T |
| Uji McNemar eksak untuk membandingkan dua model | Rainio dkk. 2024 (McNemar dibahas) | Sama | T |
| Koreksi Holm untuk perbandingan berganda | Rainio dkk. 2024 membahas Bonferroni dan menyebutnya terlalu ketat. | Modifikasi: Holm dipakai karena mengendalikan galat yang sama dengan Bonferroni, tetapi kurang ketat. | T |
| Confusion matrix | Rainio dkk. 2024 | Sama | T |
| Kandidat salah label (prediksi salah dengan keyakinan ≥ 0,90) | Northcutt dkk. 2021, *JAIR*, [10.1613/jair.1.12125](https://doi.org/10.1613/jair.1.12125) (confident learning). Lalani dkk. 2021 (kode Death di MAUDE tidak andal). | Modifikasi: kita memakai ambang keyakinan sederhana, bukan estimasi joint distribution, dan hasilnya hanya kandidat untuk ditelaah manusia. | A |
| Alasan kesalahan berbasis aturan (8 kategori) | Percha dkk. 2021 juga melakukan analisis kesalahan rinci (misalnya riwayat penyakit dan singkatan). | Modifikasi: kategorinya ditentukan dari data MAUDE MRI. | A |

## Celah yang masih terbuka

1. **AdamW, warmup + linear decay, dan early stopping pada macro-F1 val** belum punya rujukan ≥2021 yang sudah dicek. Untuk laporan, cukup disebut sebagai konfigurasi fine-tuning standar, atau cari dulu satu paper fine-tuning biomedis yang menyebut ketiganya.
2. **Ambang 10 kata dan cosine 0,9** hanya didasarkan pada data proyek. Tulis terus terang sebagai "aturan operasional" beserta angka retensinya.
3. **Paper berstatus "M"** (Bayer 2023, Lewis 2020, Yin 2019, Agarwal 2026) baru dicek metadatanya. Bila kalimat yang mengutipnya menyebut isi yang spesifik, baca dulu teks penuhnya.
