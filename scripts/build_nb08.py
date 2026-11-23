"""Generator notebook 08: eksperimen lengkap train/val/test (dijalankan di Kaggle GPU).

    python scripts/build_nb08.py          -> notebooks/08_eksperimen_lengkap.ipynb (TANPA keluaran)

Notebook mandiri: membaca data mentah openFDA (maude_mri_LNH.csv), lalu cleaning, EDA, praproses,
split train/val/test per grup, augmentasi back-translation, penanganan ketidakseimbangan (tanpa / bobot /
SMOTE atau ROS), model ML, DL (BiLSTM), DL hybrid (BiomedBERT + BiLSTM + attention), DL hybrid + pencarian
hiperparameter, BART-large-MNLI fine-tune (bukan zero-shot), evaluasi test, confusion matrix, dan analisis
kesalahan. Mode uji kilat lokal: set NB08_SMOKE=1 (subset kecil, model mini, 1 epoch).
Generator ini MENIMPA notebook beserta keluarannya.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
M, C = "md", "code"
CELLS = []


def md(s):
    CELLS.append((M, s.strip("\n")))


def code(s):
    CELLS.append((C, s.strip("\n")))


# =====================================================================================
md(r'''
# 08 · Eksperimen lengkap: cleaning → EDA → praproses → train/val/test → 5 keluarga model → analisis kesalahan

**Tugas:** klasifikasi otomatis jenis insiden keselamatan MRI (FDA MAUDE, kode produk LNH) dari narasi laporan (`event_text`) ke **4 kelas** `event_type`: Injury, Malfunction, Death, Other.

**Alur notebook**

| Bagian | Isi | Keputusan utama |
|---|---|---|
| 1 | Setup | seed tetap, GPU Kaggle |
| 2 | Cleaning | 4 kelas, narasi ≥ 10 kata, buang duplikat persis & mirip (cosine ≥ 0,9), grup kemiripan |
| 3 | EDA | sebaran kelas, panjang narasi, kelas per tahun, frasa template, grup duplikat |
| 4 | Praproses | huruf kecil, hapus kata pembocor label; dua jalur teks (`text_clean` untuk TF-IDF, `text_tf` untuk model neural) |
| 5 | Split train/val/test 70/15/15 | per grup kemiripan (laporan kembar tidak menyeberang split) |
| 6 | Augmentasi | back-translation en→de→en, **hanya data train**, kelas Death & Other |
| 7 | Desain eksperimen | 3 penanganan ketidakseimbangan × 2 augmentasi = **6 skenario** per model |
| 8 | ML | TF-IDF + Naive Bayes / LR / SVM / Random Forest, TF-IDF karakter + SVM |
| 9 | DL | BiLSTM dari nol (embedding dilatih sendiri) |
| 10 | DL hybrid | BiomedBERT (fine-tune) + BiLSTM + attention pooling + [CLS] |
| 11 | DL hybrid + hiperparameter | pencarian acak hiperparameter, dipilih dengan val |
| 12 | BART-large-MNLI fine-tune | dilatih dengan seluruh data train (bukan zero-shot) |
| 13 | Perbandingan & uji signifikansi | dipilih dengan **val**, dilaporkan di **test**; McNemar + Holm |
| 14 | Confusion matrix | model terbaik per keluarga |
| 15 | Analisis kesalahan | label asli vs prediksi, alasan salah, kandidat salah label |
| 16 | Kesimpulan | dihitung otomatis dari hasil run ini |

**Aturan evaluasi:** data **test hanya dipakai sekali di akhir**. Epoch, skenario, dan hiperparameter dipilih dengan macro-F1 **validasi**. Metrik utama **macro-F1** (keempat kelas berbobot sama, sehingga Death yang hanya ±1% tidak tenggelam).
''')

# =====================================================================================
md("## 1. Setup")
code(r'''
import os, re, json, time, glob, math, random, warnings, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import sklearn, scipy, transformers
from IPython.display import display
from scipy.stats import binom
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imbalanced-learn"], check=True)
    from imblearn.over_sampling import SMOTE
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          MarianMTModel, MarianTokenizer, get_linear_schedule_with_warmup)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

SMOKE = os.environ.get("NB08_SMOKE") == "1"      # uji kilat lokal (subset kecil, model mini)
DI_KAGGLE = Path("/kaggle/working").exists()
SEED = 42
KELAS = ["Injury", "Malfunction", "Death", "Other"]
LAB = {k: i for i, k in enumerate(KELAS)}
dev = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()

calon = glob.glob("/kaggle/input/**/maude_mri_LNH.csv", recursive=True) + ["../data/raw/maude_mri_LNH.csv"]
DATA = next(p for p in calon if Path(p).exists())
OUT = Path("/kaggle/working") if DI_KAGGLE else Path("../outputs/nb08" + ("_smoke" if SMOKE else ""))
FIG, TAB, PRED = OUT / "figures", OUT / "tables", OUT / "predictions"
for p in (FIG, TAB, PRED):
    p.mkdir(parents=True, exist_ok=True)

# nama model Hugging Face (mode SMOKE memakai model mini acak agar kode teruji tanpa unduhan besar)
HF_HYBRID = "hf-internal-testing/tiny-random-bert" if SMOKE else "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
HF_BART = "hf-internal-testing/tiny-random-bart" if SMOKE else "facebook/bart-large-mnli"
HF_MT = ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en")

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
WAKTU = {}
print("data:", DATA, "| keluaran:", OUT.resolve(), "| device:", dev,
      torch.cuda.get_device_name(0) if dev == "cuda" else "", "| SMOKE:", SMOKE)
''')
code(r'''
def bar_labels(ax, fmt="{:.0f}", fontsize=8, padding=2):
    """Angka di atas setiap batang (wajib untuk semua bar chart proyek ini)."""
    for cont in ax.containers:
        ax.bar_label(cont, labels=[fmt.format(v) for v in cont.datavalues], fontsize=fontsize, padding=padding)

def simpan_fig(nama):
    plt.tight_layout(); plt.savefig(FIG / f"{nama}.png", dpi=130, bbox_inches="tight"); plt.show()

def skor(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {"macro_f1": f1_score(y_true, y_pred, average="macro", labels=range(4), zero_division=0),
           "akurasi": accuracy_score(y_true, y_pred)}
    f1 = f1_score(y_true, y_pred, average=None, labels=range(4), zero_division=0)
    pr = precision_score(y_true, y_pred, average=None, labels=range(4), zero_division=0)
    rc = recall_score(y_true, y_pred, average=None, labels=range(4), zero_division=0)
    for i, k in enumerate(KELAS):
        out[f"f1_{k}"], out[f"precision_{k}"], out[f"recall_{k}"] = f1[i], pr[i], rc[i]
    return out

# kata yang sama dengan nama label atau kalimat template pabrikan; kata yang MENCERITAKAN kejadian
# (died, expired, passed away, burn) sengaja dipertahankan
POLA_BOCOR = re.compile(r"\b(injur\w*|serious\w*|harm\w*|malfunction\w*|deaths?)\b")
def norm_teks(t):            # untuk deteksi duplikat
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", t.lower())).strip()
def ke_text_clean(t):        # jalur TF-IDF: huruf saja + kata pembocor dihapus
    return re.sub(r"\s+", " ", POLA_BOCOR.sub(" ", norm_teks(t))).strip()
def ke_text_tf(t):           # jalur neural: angka & tanda baca dipertahankan, kata pembocor dihapus
    return re.sub(r"\s+", " ", POLA_BOCOR.sub(" ", t.lower())).strip()
''')

# =====================================================================================
md(r'''
## 2. Cleaning

| Langkah | Alasan |
|---|---|
| Ambil 4 kelas, buang label kosong | target sesuai kode pelapor `event_type` |
| Fitur hanya `event_text` | `patient_outcome` hampir identik dengan label Death (kebocoran); `mfr_narrative` ditulis setelah investigasi pabrikan |
| Narasi ≥ 10 kata | narasi yang lebih pendek tidak berisi deskripsi kejadian |
| Buang duplikat persis | laporan template pabrikan disalin berulang (terutama Malfunction) |
| Buang duplikat mirip (cosine TF-IDF ≥ 0,9) dalam kelas yang sama | model jangan sekadar menghafal template; yang dipertahankan laporan paling awal |
| Grup kemiripan lintas kelas | dipakai saat split agar laporan kembar tidak masuk train dan test sekaligus |
''')
code(r'''
t0 = time.time()
kol = ["mdr_report_key", "date_received", "event_type", "manufacturer", "event_text"]
df = pd.read_csv(DATA, keep_default_na=False, dtype={"mdr_report_key": str, "date_received": str})[kol]
print("data mentah:", len(df), "laporan"); display(df.event_type.replace("", "(kosong)").value_counts().rename("jumlah").to_frame())
tahap = {}
d = df[df.event_type.isin(KELAS)].copy()
tahap["0. 4 kelas berlabel"] = d.event_type.value_counts()
d["n_kata"] = d.event_text.str.split().str.len()
gugur = d[d.n_kata < 10]
d = d[d.n_kata >= 10]
tahap["1. narasi >= 10 kata"] = d.event_type.value_counts()
print("gugur oleh filter 10 kata per kelas:", gugur.event_type.value_counts().to_dict())
d["norm"] = d.event_text.map(norm_teks)
d = d.sort_values(["date_received", "mdr_report_key"]).drop_duplicates("norm").reset_index(drop=True)
tahap["2. buang duplikat persis"] = d.event_type.value_counts()
''')
code(r'''
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X = vec.fit_transform(d.norm)
S = (X @ X.T).tocsr(); S.data[S.data < 0.9] = 0; S.eliminate_zeros()
n_grup, grup = connected_components(S, directed=False)       # grup lintas kelas
d["grup"] = grup
ukuran_grup_awal = d.groupby("grup").size()
lab = d.event_type.values; simpan = np.ones(len(d), bool)
for i in range(len(d)):                                        # dedup mirip dalam kelas, urut waktu
    if simpan[i]:
        for j in S.getrow(i).indices:
            if j > i and lab[j] == lab[i]:
                simpan[j] = False
d = d[simpan].reset_index(drop=True).rename(columns={"event_type": "label"})
tahap["3. buang duplikat mirip"] = d.label.value_counts()
tab = pd.DataFrame(tahap).T[KELAS].fillna(0).astype(int); tab["Total"] = tab.sum(1)
tab.to_csv(TAB / "02_tahap_pembersihan.csv"); display(tab)
ax = tab[KELAS].plot.bar(figsize=(10, 4), rot=0, width=0.85)
bar_labels(ax, fontsize=7); ax.set_title("Jumlah laporan per tahap cleaning"); ax.set_ylabel("laporan")
simpan_fig("02_tahap_pembersihan")
campur = d.groupby("grup").label.nunique()
print(f"data bersih: {len(d)} laporan | {d.grup.nunique()} grup | grup multi-kelas: {(campur > 1).sum()}")
WAKTU["cleaning"] = time.time() - t0
''')

# =====================================================================================
md(r'''
## 3. Exploratory Data Analysis (EDA)

EDA dijalankan pada data bersih untuk menjawab lima pertanyaan yang menentukan keputusan praproses dan model.
''')
md("### 3.1 Seberapa timpang kelasnya?")
code(r'''
n = d.label.value_counts()[KELAS]
fig, ax = plt.subplots(figsize=(7, 3.8))
b = ax.bar(KELAS, n.values, color=sns.color_palette("Set2", 4))
ax.bar_label(b, labels=[f"{v} ({v / n.sum() * 100:.1f}%)" for v in n.values], fontsize=9, padding=2)
ax.set_title("Sebaran kelas data bersih"); ax.set_ylabel("laporan"); ax.set_ylim(0, n.max() * 1.15)
simpan_fig("03_sebaran_kelas")
print(f"rasio kelas terbesar : terkecil = {n.max() / n.min():.0f} : 1")
''')
md("### 3.2 Apakah panjang narasi berbeda antarkelas?")
code(r'''
d["n_kata"] = d.event_text.str.split().str.len()
display(d.groupby("label").n_kata.describe()[["count", "mean", "50%", "min", "max"]].loc[KELAS].round(1))
fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(data=d, x="label", y="n_kata", order=KELAS, ax=ax, palette="Set2", showfliers=False)
ax.axhline(10, ls="--", c="r", lw=1, label="batas filter 10 kata"); ax.legend()
ax.set_title("Panjang narasi per kelas (tanpa pencilan)"); ax.set_ylabel("jumlah kata")
simpan_fig("03_panjang_narasi")
''')
md("### 3.3 Apakah komposisi kelas berubah menurut tahun penerimaan?")
code(r'''
d["tahun"] = pd.to_datetime(d.date_received, format="%Y%m%d", errors="coerce").dt.year
per_tahun = pd.crosstab(d.tahun, d.label)[KELAS]
fig, ax = plt.subplots(figsize=(11, 4))
for k in KELAS:
    ax.plot(per_tahun.index, per_tahun[k], marker="o", ms=3, label=k)
ax.set_yscale("symlog"); ax.legend(); ax.set_title("Jumlah laporan per tahun penerimaan (skala log)"); ax.set_xlabel("tahun")
simpan_fig("03_kelas_per_tahun")
o = d[d.label == "Other"]
print(f"Other diterima s.d. 2010: {(o.tahun <= 2010).sum()}/{len(o)} -> kelas Other sebagian besar artefak pengodean lama")
''')
md("### 3.4 Frasa apa yang menandai tiap kelas? (template vs kata kejadian)")
code(r'''
frasa = {"serious injury": r"serious injur", "no injury": r"no injur", "no patient": r"no patient",
         "malfunction": r"malfunction", "death": r"\bdeath", "died": r"\bdied\b", "expired": r"\bexpired\b",
         "burn*": r"\bburn\w*", "pain": r"\bpain\b", "quench": r"\bquench"}
prev = pd.DataFrame({k: [d[d.label == k].norm.str.contains(p).mean() * 100 for p in frasa.values()] for k in KELAS},
                    index=list(frasa)).round(1)
prev.to_csv(TAB / "03_prevalensi_frasa.csv")
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.heatmap(prev, annot=True, fmt=".1f", cmap="Blues", ax=ax, cbar_kws={"label": "% laporan"})
ax.set_title("% laporan per kelas yang memuat frasa"); simpan_fig("03_prevalensi_frasa")
''')
md("### 3.5 Seberapa besar masalah duplikat/template?")
code(r'''
top = ukuran_grup_awal.sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(9, 3.6))
b = ax.bar([f"g{g}" for g in top.index], top.values, color="#4C72B0"); bar_labels(ax)
ax.set_title("10 grup kemiripan terbesar (sebelum dedup mirip)"); ax.set_ylabel("laporan dalam grup")
simpan_fig("03_grup_terbesar")
print("grup berisi >1 laporan sebelum dedup mirip:", (ukuran_grup_awal > 1).sum(),
      "| laporan di dalamnya:", ukuran_grup_awal[ukuran_grup_awal > 1].sum())
''')
md(r'''
### 3.6 Kesimpulan EDA → keputusan

| Temuan | Keputusan |
|---|---|
| Kelas sangat timpang (Death ±1%) | metrik **macro-F1**; bandingkan **tanpa penanganan vs bobot kelas vs SMOTE/ROS**; split bertingkat (stratified) |
| Frasa template (`serious injury`, `malfunction`) menandai gaya pelapor, bukan kejadian | hapus kata pembocor label; kata kejadian (`died`, `burn`) dipertahankan |
| Banyak laporan kembar/template | dedup + **split per grup kemiripan** |
| Other hampir semuanya laporan lama | kelas Other dipertahankan (sesuai kode pelapor), dibahas di analisis kesalahan |
| Narasi bervariasi panjangnya | batas 256 token untuk transformer, 300 token untuk BiLSTM |
''')

# =====================================================================================
md(r'''
## 4. Praproses

Dua jalur teks:
* **`text_clean`** (TF-IDF/ML): huruf kecil → hanya huruf a–z → hapus kata pembocor → rapikan spasi. Stopword Inggris dibuang hanya pada TF-IDF kata. **Tanpa stemming/lemmatisasi**: TF-IDF karakter (3–5 gram) sudah menangkap variasi bentuk kata.
* **`text_tf`** (BiLSTM, hybrid, BART): huruf kecil → hapus kata pembocor; angka dan tanda baca dipertahankan karena tokenizer subkata memakainya sebagai konteks.
''')
code(r'''
d["text_clean"] = d.event_text.map(ke_text_clean)
d["text_tf"] = d.event_text.map(ke_text_tf)
contoh = d.sample(3, random_state=SEED)
for _, r in contoh.iterrows():
    print(f"[{r.label}] ASLI      : {r.event_text[:220]}\n         text_clean: {r.text_clean[:220]}\n         text_tf   : {r.text_tf[:220]}\n")
print("rata-rata kata pembocor yang dihapus per laporan:", round(d.event_text.str.lower().str.count(POLA_BOCOR.pattern).mean(), 2))
''')

# =====================================================================================
md(r'''
## 5. Split train / validation / test (70/15/15) per grup kemiripan

`StratifiedGroupKFold` 20 lipatan: 3 lipatan → test, 3 → val, 14 → train. Stratified menjaga proporsi kelas, group menjaga laporan kembar tetap di satu bagian.

**Reprodusibilitas.**
- Partisi hasil `StratifiedGroupKFold` bisa berbeda antar-lingkungan, misalnya bila versi pustaka di Kaggle berbeda dengan versi lokal. Run Kaggle 23 Nov 2026 menghasilkan train/val/test 2.536/534/534, sedangkan run lokal menghasilkan 2.530/537/537.
- Karena itu, bila ada file split beku (`split_nb08_beku.csv`: `mdr_report_key, split`), notebook memakai file itu.
- Setiap run juga menyimpan manifest partisinya ke `tables/05_manifest_split.csv`, dan versi pustakanya ke `run_info.json`.
''')
code(r'''
beku = glob.glob("/kaggle/input/**/split_nb08_beku.csv", recursive=True) + ["../data/processed/split_nb08_beku.csv"]
beku = next((p for p in beku if Path(p).exists()), None)
if beku:
    sb = pd.read_csv(beku, dtype=str).set_index("mdr_report_key").split
    assert d.mdr_report_key.isin(sb.index).all(), "split beku tidak mencakup semua laporan"
    d["split"] = d.mdr_report_key.map(sb).values
    print("memakai split beku:", beku)
else:
    f = np.empty(len(d), int)
    for i, (_, te) in enumerate(StratifiedGroupKFold(20, shuffle=True, random_state=SEED).split(d, d.label, d.grup)):
        f[te] = i
    d["split"] = np.where(f < 3, "test", np.where(f < 6, "val", "train"))
    print("split dihitung ulang (tidak ada file split beku)")
assert (d.groupby("grup").split.nunique() == 1).all(), "ada grup yang menyeberang split"
d[["mdr_report_key", "grup", "label", "split"]].to_csv(TAB / "05_manifest_split.csv", index=False)
if SMOKE:   # subset kecil, tetap memuat keempat kelas
    d = pd.concat([g.sample(frac=1, random_state=SEED).groupby("label").head(n)
                   for (s, g), n in zip(d.groupby("split"), [12, 40, 12])]).reset_index(drop=True)
ct = pd.crosstab(d.split, d.label)[KELAS].loc[["train", "val", "test"]]; ct["Total"] = ct.sum(1)
ct.to_csv(TAB / "05_split.csv"); display(ct)
print("grup yang menyeberang split:", (d.groupby("grup").split.nunique() > 1).sum())
ax = ct[KELAS].plot.bar(figsize=(8, 3.8), rot=0, width=0.85); bar_labels(ax)
ax.set_title("Jumlah laporan per split dan kelas"); ax.set_ylabel("laporan"); simpan_fig("05_split")
tr, va, te = (d[d.split == s].reset_index(drop=True) for s in ["train", "val", "test"])
y_tr, y_va, y_te = (x.label.map(LAB).values for x in (tr, va, te))
''')

# =====================================================================================
md(r'''
## 6. Augmentasi teks: back-translation (hanya data train)

Teks train kelas **Death** dan **Other** diterjemahkan ke bahasa Jerman lalu kembali ke Inggris (MarianMT, `Helsinki-NLP/opus-mt`). Val dan test **tidak pernah** diaugmentasi. Kata pembocor dihapus ulang setelah terjemahan karena terjemahan bisa memunculkannya kembali.
''')
code(r'''
t0 = time.time()
KELAS_AUG = ["Death", "Other"]
sumber = tr[tr.label.isin(KELAS_AUG)].reset_index(drop=True)

def pecah_kalimat(t):
    s = [x.strip() for x in re.split(r"(?<=[.!?])\s+", t.lower()) if x.strip()]
    return [x[0].upper() + x[1:] for x in s]

def terjemah(kal, nama, bs=32):
    tok = MarianTokenizer.from_pretrained(nama); mdl = MarianMTModel.from_pretrained(nama).to(dev).eval()
    hasil = []
    with torch.inference_mode():
        for i in range(0, len(kal), bs):
            enc = tok(kal[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
            hasil += tok.batch_decode(mdl.generate(**enc, num_beams=4, max_new_tokens=300), skip_special_tokens=True)
    del mdl; torch.cuda.empty_cache()
    return hasil

kal = [pecah_kalimat(t) for t in sumber.event_text]
rata = [k for ks in kal for k in ks]; idx = np.repeat(np.arange(len(kal)), [len(k) for k in kal])
if SMOKE:   # tanpa unduh model terjemahan: urutan kalimat dibalik sebagai pengganti
    balik = [k for ks in kal for k in ks[::-1]]
else:
    balik = terjemah(terjemah(rata, HF_MT[0]), HF_MT[1])
teks_aug = [" ".join(b for b, k in zip(balik, idx) if k == j) for j in range(len(sumber))]
aug = sumber[["mdr_report_key", "label", "event_text"]].copy()
aug["text_aug"] = teks_aug
aug["bocor_muncul"] = aug.text_aug.str.lower().str.contains(POLA_BOCOR.pattern)
aug["text_tf"] = aug.text_aug.map(ke_text_tf); aug["text_clean"] = aug.text_aug.map(ke_text_clean)
aug.to_csv(TAB / "06_augmentasi.csv", index=False)
v = TfidfVectorizer().fit(pd.concat([aug.event_text, aug.text_aug]))
aug["cos_asli"] = np.asarray((v.transform(aug.event_text).multiply(v.transform(aug.text_aug))).sum(1)).ravel()
print("teks augmentasi:", aug.label.value_counts().to_dict(), "| memunculkan kata pembocor lagi:", aug.bocor_muncul.sum(),
      "| median cosine ke teks asli:", round(aug.cos_asli.median(), 3))
for _, r in aug.head(2).iterrows():
    print(f"\n[{r.label}] ASLI: {r.event_text[:250]}\n     AUG : {r.text_aug[:250]}")
WAKTU["augmentasi"] = time.time() - t0
''')

# =====================================================================================
md(r'''
## 7. Desain eksperimen

Setiap model dijalankan pada **6 skenario** = 3 penanganan ketidakseimbangan × 2 augmentasi:

| Penanganan | ML (TF-IDF) | DL / hybrid / BART |
|---|---|---|
| `tanpa` | tanpa penanganan | cross-entropy biasa |
| `bobot` | `class_weight="balanced"` (NB: `sample_weight`) | cross-entropy berbobot `n / (4 · n_kelas)` |
| `resample` | **SMOTE** pada vektor TF-IDF data train | **ROS** (oversampling acak lewat sampler seimbang) |

SMOTE membuat sampel sintetis dengan interpolasi vektor fitur. Pada model neural, masukannya urutan token (bilangan bulat indeks kata), sehingga interpolasi tidak bermakna (token 512,4 bukan kata). Karena itu padanan SMOTE di model neural adalah **ROS**.

| Augmentasi | Data train |
|---|---|
| `tanpa_aug` | teks asli |
| `aug_bt` | teks asli + back-translation (Death & Other) |

**Pemilihan:** untuk setiap model, epoch terbaik dipilih dengan macro-F1 val. Skenario terbaik setiap keluarga model juga dipilih dengan val. Test dilaporkan untuk semua run sebagai bahan analisis, tetapi **tidak** dipakai memilih.
''')
code(r'''
IMB = ["tanpa", "bobot", "resample"]
AUG = ["tanpa_aug", "aug_bt"]
HASIL, PROBA = [], {}

def data_train(aug_mode, kolom):
    if aug_mode == "aug_bt":
        teks = pd.concat([tr[kolom], aug[kolom]]).tolist(); y = np.r_[y_tr, aug.label.map(LAB).values]
    else:
        teks = tr[kolom].tolist(); y = y_tr.copy()
    return teks, y

def catat(keluarga, model, imb, aug_mode, p_va, p_te, detik, extra=None):
    """p_va/p_te: matriks skor (n x 4) atau label prediksi (n,)."""
    pv = p_va.argmax(1) if np.ndim(p_va) == 2 else np.asarray(p_va)
    pt = p_te.argmax(1) if np.ndim(p_te) == 2 else np.asarray(p_te)
    sv, st = skor(y_va, pv), skor(y_te, pt)
    nama_imb = {"resample": "SMOTE" if keluarga == "ML" else "ROS"}.get(imb, imb)
    baris = {"keluarga": keluarga, "model": model, "imbalance": nama_imb, "augmentasi": aug_mode,
             "val_macro_f1": sv["macro_f1"], **{f"test_{k}": v for k, v in st.items()}, "detik": detik, **(extra or {})}
    HASIL.append(baris)
    kunci = f"{keluarga}|{model}|{nama_imb}|{aug_mode}"
    PROBA[kunci] = (p_va, p_te)
    out = pd.DataFrame({"mdr_report_key": te.mdr_report_key, "y_true": te.label, "y_pred": [KELAS[i] for i in pt]})
    if np.ndim(p_te) == 2:
        for i, k in enumerate(KELAS):
            out[f"p_{k}"] = p_te[:, i]
    out.to_csv(PRED / (re.sub(r"[^\w\-]+", "_", kunci) + ".csv"), index=False)
    print(f"{kunci:70s} val {sv['macro_f1']:.3f} | test {st['macro_f1']:.3f} | F1 Death test {st['f1_Death']:.2f} | {detik:.0f}s", flush=True)
''')

# =====================================================================================
md(r'''
## 8. Machine Learning: TF-IDF + model klasik

| Model | Representasi | Hiperparameter (tetap, dari eksperimen 5-fold sebelumnya) |
|---|---|---|
| Naive Bayes | TF-IDF kata 1–2 gram | ComplementNB α = 0,3 |
| Logistic Regression | TF-IDF kata 1–2 gram | C = 10 |
| SVM linear | TF-IDF kata 1–2 gram | C = 0,5 |
| Random Forest | TF-IDF kata 1–2 gram | 500 pohon, min_samples_leaf = 2 |
| SVM linear | TF-IDF **karakter** 3–5 gram | C = 0,5 |
''')
code(r'''
def vek_kata(): return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True, stop_words="english")
def vek_char(): return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=200000)
ML = {
    "TF-IDF kata + Naive Bayes": (vek_kata, lambda cw: ComplementNB(alpha=0.3)),
    "TF-IDF kata + Logistic Regression": (vek_kata, lambda cw: LogisticRegression(C=10, class_weight=cw, max_iter=3000)),
    "TF-IDF kata + SVM linear": (vek_kata, lambda cw: LinearSVC(C=0.5, class_weight=cw, random_state=SEED)),
    "TF-IDF kata + Random Forest": (vek_kata, lambda cw: RandomForestClassifier(500, class_weight=cw and "balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=SEED)),
    "TF-IDF karakter + SVM linear": (vek_char, lambda cw: LinearSVC(C=0.5, class_weight=cw, random_state=SEED)),
}
def skor_ml(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    z = clf.decision_function(X); z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)

t_ml = time.time()
for nama, (vek, buat) in ML.items():
    for aug_mode in AUG:
        teks, y = data_train(aug_mode, "text_clean")
        v = vek(); Xtr = v.fit_transform(teks); Xva, Xte = v.transform(va.text_clean), v.transform(te.text_clean)
        for imb in IMB:
            t0 = time.time(); Xi, yi = Xtr, y
            if imb == "resample":
                k = max(1, min(5, np.bincount(y, minlength=4)[np.bincount(y, minlength=4) > 0].min() - 1))
                Xi, yi = SMOTE(random_state=SEED, k_neighbors=k).fit_resample(Xtr, y)
            clf = buat("balanced" if imb == "bobot" else None)
            if "Naive Bayes" in nama and imb == "bobot":
                clf.fit(Xi, yi, sample_weight=compute_sample_weight("balanced", yi))
            else:
                clf.fit(Xi, yi)
            catat("ML", nama, imb, aug_mode, skor_ml(clf, Xva), skor_ml(clf, Xte), time.time() - t0)
WAKTU["ML"] = time.time() - t_ml
''')

# =====================================================================================
md(r'''
## 9–12. Model neural: kerangka latih bersama

Semua model neural memakai loop latih yang sama, sehingga perbandingannya adil:
* AdamW, *warm-up* 10% + *linear decay*, *gradient clipping* 1,0, *mixed precision* (fp16) di GPU.
* Setiap epoch dinilai di val; **early stopping** dengan kesabaran 2 epoch; prediksi val & test diambil dari **epoch dengan macro-F1 val terbaik**.
* Train loss dan val loss dicatat per epoch (kurva di bagian 13).
''')
code(r'''
EPOCH_DL, EPOCH_HY, EPOCH_BART = (1, 1, 1) if SMOKE else (15, 5, 4)

def batch_index(n, bs, rng, y=None, imb="tanpa"):
    if imb == "resample":                  # ROS: tiap kelas berpeluang sama, panjang epoch tetap n
        cnt = np.bincount(y, minlength=4); p = 1.0 / np.maximum(cnt[y], 1); p /= p.sum()
        idx = rng.choice(n, n, replace=True, p=p)
    else:
        idx = rng.permutation(n)
    return [idx[i:i + bs] for i in range(0, n, bs)]

def prediksi(model, items, collate, bs):
    model.eval(); urut = np.argsort([len(x["input_ids"]) for x in items]); P = np.zeros((len(items), 4), np.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
        for i in range(0, len(items), bs):
            b = urut[i:i + bs]
            P[b] = torch.softmax(model(**collate([items[j] for j in b])).float(), -1).cpu().numpy()
    return P

def latih(model, groups, tr_items, y, collate, imb, epochs, bs, patience=2, label=""):
    rng = np.random.default_rng(SEED); n = len(y); cnt = np.bincount(y, minlength=4)
    w = torch.tensor(n / (4 * np.maximum(cnt, 1)), dtype=torch.float, device=dev) if imb == "bobot" else None
    lossf, loss_val = nn.CrossEntropyLoss(weight=w), nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    langkah = epochs * math.ceil(n / bs); sch = get_linear_schedule_with_warmup(opt, int(0.1 * langkah), langkah)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    best, simpan, hist, tunggu = -1, None, [], 0
    for ep in range(epochs):
        model.train(); tl = []
        for b in batch_index(n, bs, rng, y, imb):
            x = collate([tr_items[j] for j in b]); yt = torch.tensor(y[b], device=dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = lossf(model(**x).float(), yt)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step(); tl.append(loss.item())
        Pv = prediksi(model, VA_ITEMS, collate, bs * 2)
        vl = loss_val(torch.tensor(np.log(Pv + 1e-9)), torch.tensor(y_va)).item()
        f1v = f1_score(y_va, Pv.argmax(1), average="macro", labels=range(4), zero_division=0)
        hist.append({"model": label, "epoch": ep + 1, "train_loss": np.mean(tl), "val_loss": vl, "val_macro_f1": f1v})
        print(f"   {label} epoch {ep + 1}: train loss {np.mean(tl):.4f} | val loss {vl:.4f} | val macro-F1 {f1v:.4f}", flush=True)
        if f1v > best:
            best, tunggu = f1v, 0
            simpan = (Pv, prediksi(model, TE_ITEMS, collate, bs * 2), ep + 1)
        else:
            tunggu += 1
            if tunggu >= patience:
                break
    return simpan, hist

RIWAYAT = []
''')

# ---------------------------------------------------------------- DL
md(r'''
## 9. Deep Learning: BiLSTM dari nol

Embedding kata (128 dim) **dilatih dari nol** dari data train → BiLSTM 2 arah (128 unit) → gabungan *max pooling* + *mean pooling* → dropout 0,3 → linear 4 kelas. Kosakata: token yang muncul ≥ 2 kali di train; panjang maks 300 token.
''')
code(r'''
def tokenisasi(t): return re.findall(r"[a-z]+|\d+", t)

class BiLSTM(nn.Module):
    def __init__(self, vocab, emb=128, hid=128, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop); self.out = nn.Linear(4 * hid, 4)
    def forward(self, input_ids, attention_mask):
        h, _ = self.lstm(self.drop(self.emb(input_ids)))
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        mx = h.masked_fill(m == 0, -1e4).max(1).values
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(self.drop(torch.cat([mx, mean], 1)))

def collate_ids(items):
    L = max(len(x["input_ids"]) for x in items)
    ids = torch.zeros(len(items), L, dtype=torch.long)
    for i, x in enumerate(items):
        ids[i, :len(x["input_ids"])] = torch.tensor(x["input_ids"])
    return {"input_ids": ids.to(dev), "attention_mask": (ids != 0).long().to(dev)}

t_dl = time.time()
for aug_mode in AUG:
    teks, y = data_train(aug_mode, "text_tf")
    from collections import Counter
    frek = Counter(w for t in teks for w in tokenisasi(t))
    vocab = {w: i + 2 for i, w in enumerate(w for w, c in frek.items() if c >= 2)}   # 0 = pad, 1 = OOV
    ke_item = lambda t: {"input_ids": [vocab.get(w, 1) for w in tokenisasi(t)][:300] or [1]}
    tr_items = [ke_item(t) for t in teks]
    VA_ITEMS, TE_ITEMS = [ke_item(t) for t in va.text_tf], [ke_item(t) for t in te.text_tf]
    for imb in IMB:
        set_seed(); t0 = time.time()
        model = BiLSTM(len(vocab) + 2).to(dev)
        (pv, pt, ep), h = latih(model, [{"params": model.parameters(), "lr": 2e-3}], tr_items, y, collate_ids, imb,
                                EPOCH_DL, 32, patience=3, label=f"BiLSTM|{imb}|{aug_mode}")
        RIWAYAT += h
        catat("DL", "BiLSTM", imb, aug_mode, pv, pt, time.time() - t0, {"epoch_terbaik": ep, "vocab": len(vocab)})
        del model; torch.cuda.empty_cache()
WAKTU["DL"] = time.time() - t_dl
''')

# ---------------------------------------------------------------- Hybrid
md(r'''
## 10. Deep Learning hybrid: BiomedBERT + BiLSTM + attention

```
narasi ─► BiomedBERT (encoder kontekstual biomedis, di-fine-tune)
             ├─► [CLS] ─────────────────────────────┐
             └─► BiLSTM ─► attention pooling ───────┤ concat ─► Dropout ─► Linear(512)+GELU ─► Dropout ─► Linear(4)
```

* **Encoder:** `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (PubMedBERT). Model ini dipilih karena pada eksperimen embedding sebelumnya PubMedBERT memberi macro-F1 tertinggi di antara tiga encoder.
* **BiLSTM + attention:** membaca urutan representasi token dan memberi bobot pada token yang paling informatif (misalnya *expired*, *burn*). Ini melengkapi [CLS], yang hanya meringkas satu posisi.
* Learning rate berbeda: encoder `3e-5`, cabang baru `1e-3`. Panjang maks 256 token, batch 16, maks 5 epoch.
''')
code(r'''
class Hybrid(nn.Module):
    def __init__(self, hf, hid=256, drop=0.3):
        super().__init__()
        self.enc = AutoModel.from_pretrained(hf); H = self.enc.config.hidden_size
        self.lstm = nn.LSTM(H, hid, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2 * hid, 1); self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(H + 2 * hid, 512), nn.GELU(), nn.Dropout(drop), nn.Linear(512, 4))
    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        l, _ = self.lstm(self.drop(h))
        s = self.att(l).squeeze(-1).float().masked_fill(attention_mask == 0, -1e4)
        a = torch.softmax(s, -1).unsqueeze(-1)
        pooled = (l.float() * a).sum(1)
        return self.head(self.drop(torch.cat([h[:, 0].float(), pooled], 1)))

tok_hy = AutoTokenizer.from_pretrained(HF_HYBRID)
def items_hf(tok, teks, max_len):
    enc = tok(list(teks), truncation=True, max_length=max_len)
    return [{"input_ids": a, "attention_mask": m} for a, m in zip(enc["input_ids"], enc["attention_mask"])]
def collate_hf(tok):
    def f(items):
        return {k: v.to(dev) for k, v in tok.pad(items, return_tensors="pt").items() if k in ("input_ids", "attention_mask")}
    return f

def run_hybrid(imb, aug_mode, lr_enc=3e-5, lr_head=1e-3, drop=0.3, hid=256, bs=16, epochs=EPOCH_HY, label="Hybrid", catat_ke=None):
    global VA_ITEMS, TE_ITEMS
    teks, y = data_train(aug_mode, "text_tf")
    tr_items = items_hf(tok_hy, teks, 256)
    VA_ITEMS, TE_ITEMS = items_hf(tok_hy, va.text_tf, 256), items_hf(tok_hy, te.text_tf, 256)
    set_seed(); t0 = time.time()
    model = Hybrid(HF_HYBRID, hid, drop).to(dev)
    enc_p = list(model.enc.parameters()); ids = {id(p) for p in enc_p}
    groups = [{"params": enc_p, "lr": lr_enc}, {"params": [p for p in model.parameters() if id(p) not in ids], "lr": lr_head}]
    (pv, pt, ep), h = latih(model, groups, tr_items, y, collate_hf(tok_hy), imb, epochs, bs, label=f"{label}|{imb}|{aug_mode}")
    del model; torch.cuda.empty_cache()
    return pv, pt, ep, h, time.time() - t0

NAMA_HY = "BiomedBERT + BiLSTM + attention"
t_hy = time.time()
for aug_mode in AUG:
    for imb in IMB:
        pv, pt, ep, h, dt = run_hybrid(imb, aug_mode)
        RIWAYAT += h
        catat("DL hybrid", NAMA_HY, imb, aug_mode, pv, pt, dt, {"epoch_terbaik": ep})
WAKTU["DL hybrid"] = time.time() - t_hy
''')

# ---------------------------------------------------------------- Hybrid + HP
md(r'''
## 11. DL hybrid + pencarian hiperparameter

Skenario (penanganan ketidakseimbangan + augmentasi) diambil dari run hybrid dengan **macro-F1 val tertinggi** di bagian 10. Lalu dilakukan **pencarian acak** atas ruang hiperparameter berikut. Setiap percobaan dinilai di val, dan percobaan terbaik dilaporkan di test.

| Hiperparameter | Ruang pencarian |
|---|---|
| learning rate encoder | 1e-5, 2e-5, 3e-5, 5e-5 |
| learning rate cabang baru | 5e-4, 1e-3, 2e-3 |
| dropout | 0,1 / 0,2 / 0,3 / 0,5 |
| unit BiLSTM | 128, 256 |
| batch size | 8, 16 |
''')
code(r'''
N_TRIAL = 2 if SMOKE else 8
hy = pd.DataFrame([r for r in HASIL if r["keluarga"] == "DL hybrid"]).sort_values("val_macro_f1", ascending=False).iloc[0]
imb_hp = {"ROS": "resample"}.get(hy.imbalance, hy.imbalance); aug_hp = hy.augmentasi
print(f"skenario terpilih dari bagian 10: imbalance={hy.imbalance}, augmentasi={aug_hp} (val {hy.val_macro_f1:.3f})")
RUANG = {"lr_enc": [1e-5, 2e-5, 3e-5, 5e-5], "lr_head": [5e-4, 1e-3, 2e-3], "drop": [0.1, 0.2, 0.3, 0.5], "hid": [128, 256], "bs": [8, 16]}
rng_hp = np.random.default_rng(2026); trials, terbaik = [], None
t_hp = time.time()
for t in range(N_TRIAL):
    hp = {k: v[rng_hp.integers(len(v))] for k, v in RUANG.items()}
    pv, pt, ep, h, dt = run_hybrid(imb_hp, aug_hp, label=f"HP{t + 1}", **{k: (int(v) if k in ("hid", "bs") else float(v)) for k, v in hp.items()})
    f1v = f1_score(y_va, pv.argmax(1), average="macro", labels=range(4), zero_division=0)
    trials.append({"trial": t + 1, **hp, "epoch_terbaik": ep, "val_macro_f1": f1v, "detik": dt})
    print(f"trial {t + 1}: {hp} -> val {f1v:.4f}", flush=True)
    if terbaik is None or f1v > terbaik[0]:
        terbaik = (f1v, pv, pt, ep, h, dt, hp)
tr_df = pd.DataFrame(trials); tr_df.to_csv(TAB / "11_trial_hiperparameter.csv", index=False); display(tr_df.round(4))
f1v, pv, pt, ep, h, dt, hp = terbaik
RIWAYAT += [{**x, "model": "Hybrid+HP terbaik|" + x["model"]} for x in h]
catat("DL hybrid + HP", NAMA_HY + " (tuned)", imb_hp, aug_hp, pv, pt, dt, {"epoch_terbaik": ep, "hiperparameter": json.dumps({k: float(v) for k, v in hp.items()})})
fig, ax = plt.subplots(figsize=(8, 3.5))
b = ax.bar(tr_df.trial.astype(str), tr_df.val_macro_f1, color="#55A868"); bar_labels(ax, "{:.3f}")
ax.axhline(hy.val_macro_f1, ls="--", c="r", lw=1, label=f"hybrid bawaan (val {hy.val_macro_f1:.3f})"); ax.legend(fontsize=8)
ax.set_xlabel("trial"); ax.set_ylabel("macro-F1 val"); ax.set_ylim(0, 1); ax.set_title("Pencarian hiperparameter hybrid")
simpan_fig("11_trial_hiperparameter")
WAKTU["DL hybrid + HP"] = time.time() - t_hp
''')

# ---------------------------------------------------------------- BART
md(r'''
## 12. BART-large-MNLI fine-tune (bukan zero-shot)

`facebook/bart-large-mnli` (406 juta parameter, sudah dilatih untuk inferensi bahasa alami) **dilatih ulang** pada seluruh data train sebagai klasifikasi 4 kelas. Kepala klasifikasi MNLI (3 kelas: entailment/neutral/contradiction) diganti dengan kepala baru 4 kelas; seluruh bobot encoder–decoder ikut di-fine-tune.
Learning rate 1e-5 (model besar, rentan tidak stabil), batch 8, maks 4 epoch, panjang 256 token.
''')
code(r'''
class BartCls(nn.Module):
    def __init__(self, hf):
        super().__init__()
        self.m = AutoModelForSequenceClassification.from_pretrained(hf, num_labels=4, ignore_mismatched_sizes=True,
                                                                    id2label=dict(enumerate(KELAS)), label2id=LAB)
    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

from transformers import AutoConfig
tok_bart = AutoTokenizer.from_pretrained(HF_BART)
MAXLEN_BART = min(256, AutoConfig.from_pretrained(HF_BART).max_position_embeddings)
t_bart = time.time()
for aug_mode in AUG:
    teks, y = data_train(aug_mode, "text_tf")
    tr_items = items_hf(tok_bart, teks, MAXLEN_BART)
    VA_ITEMS, TE_ITEMS = items_hf(tok_bart, va.text_tf, MAXLEN_BART), items_hf(tok_bart, te.text_tf, MAXLEN_BART)
    for imb in IMB:
        set_seed(); t0 = time.time()
        model = BartCls(HF_BART).to(dev)
        (pv, pt, ep), h = latih(model, [{"params": model.parameters(), "lr": 1e-5}], tr_items, y, collate_hf(tok_bart), imb,
                                EPOCH_BART, 8, label=f"BART|{imb}|{aug_mode}")
        RIWAYAT += h
        catat("BART fine-tune", "BART-large-MNLI fine-tune", imb, aug_mode, pv, pt, time.time() - t0, {"epoch_terbaik": ep})
        del model; torch.cuda.empty_cache()
WAKTU["BART"] = time.time() - t_bart
''')

# =====================================================================================
md(r'''
## 13. Perbandingan semua model

### 13.1 Semua run (5 keluarga × skenario)
''')
code(r'''
H = pd.DataFrame(HASIL)
H.to_csv(TAB / "13_semua_run.csv", index=False)
pd.DataFrame(RIWAYAT).to_csv(TAB / "13_riwayat_epoch.csv", index=False)
kol_tampil = ["keluarga", "model", "imbalance", "augmentasi", "val_macro_f1", "test_macro_f1", "test_akurasi",
              "test_f1_Injury", "test_f1_Malfunction", "test_f1_Death", "test_f1_Other", "test_recall_Death", "detik"]
display(H[kol_tampil].sort_values("val_macro_f1", ascending=False).round(3))
''')
md("### 13.2 Pengaruh penanganan ketidakseimbangan dan augmentasi (macro-F1 test, rata-rata per keluarga)")
code(r'''
H["imb_umum"] = H.imbalance.replace({"SMOTE": "SMOTE/ROS", "ROS": "SMOTE/ROS"})
pv_imb = H[H.keluarga != "DL hybrid + HP"].pivot_table(index="keluarga", columns="imb_umum", values="test_macro_f1", aggfunc="mean")[["tanpa", "bobot", "SMOTE/ROS"]]
pv_aug = H[H.keluarga != "DL hybrid + HP"].pivot_table(index="keluarga", columns="augmentasi", values="test_macro_f1", aggfunc="mean")[AUG]
pv_imb.to_csv(TAB / "13_efek_imbalance.csv"); pv_aug.to_csv(TAB / "13_efek_augmentasi.csv")
fig, axs = plt.subplots(1, 2, figsize=(14, 4.2))
pv_imb.plot.bar(ax=axs[0], rot=0, width=0.8); bar_labels(axs[0], "{:.3f}", fontsize=7)
axs[0].set_title("Penanganan ketidakseimbangan"); axs[0].set_ylim(0, 1); axs[0].set_ylabel("macro-F1 test (rata-rata)")
pv_aug.plot.bar(ax=axs[1], rot=0, width=0.7); bar_labels(axs[1], "{:.3f}", fontsize=7)
axs[1].set_title("Augmentasi back-translation"); axs[1].set_ylim(0, 1)
for ax in axs:
    ax.legend(fontsize=8, loc="lower right"); ax.tick_params(axis="x", labelsize=8)
simpan_fig("13_efek_imbalance_augmentasi")
# Death adalah kelas yang paling dipengaruhi penanganan ketidakseimbangan
display(H[H.keluarga != "DL hybrid + HP"].pivot_table(index="keluarga", columns="imb_umum", values="test_recall_Death", aggfunc="mean")[["tanpa", "bobot", "SMOTE/ROS"]].round(3).rename_axis("recall Death test"))
''')
md(r'''
### 13.3 Model terbaik per keluarga (dipilih dengan val, dilaporkan di test)
''')
code(r'''
best = H.sort_values("val_macro_f1", ascending=False).groupby("keluarga", sort=False).head(1)
best = best.set_index("keluarga").loc[[k for k in ["ML", "DL", "DL hybrid", "DL hybrid + HP", "BART fine-tune"] if k in set(best.keluarga)]].reset_index()
best["kunci"] = best.keluarga + "|" + best.model + "|" + best.imbalance + "|" + best.augmentasi
best[kol_tampil].round(3).to_csv(TAB / "13_terbaik_per_keluarga.csv", index=False); display(best[kol_tampil].round(3))
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(best)); w = 0.38
b1 = ax.bar(x - w / 2, best.val_macro_f1, w, label="val"); b2 = ax.bar(x + w / 2, best.test_macro_f1, w, label="test")
bar_labels(ax, "{:.3f}")
ax.set_xticks(x); ax.set_xticklabels([f"{k}\n({m[:28]})\n{i}, {a}" for k, m, i, a in zip(best.keluarga, best.model, best.imbalance, best.augmentasi)], fontsize=7)
ax.set_ylim(0, 1); ax.set_ylabel("macro-F1"); ax.set_title("Model terbaik per keluarga"); ax.legend()
simpan_fig("13_terbaik_per_keluarga")
JUARA = best.sort_values("val_macro_f1", ascending=False).iloc[0]
print(f"MODEL TERBAIK (val): {JUARA.kunci} | val {JUARA.val_macro_f1:.3f} | test {JUARA.test_macro_f1:.3f}")
''')
md(r'''
### 13.4 Uji signifikansi: McNemar (model terbaik vs model terbaik keluarga lain, data test)

McNemar menguji apakah dua model salah pada laporan yang berbeda secara bermakna (uji eksak binomial atas pasangan yang tidak sepakat). Nilai p dikoreksi Holm. Uji ini berbasis benar/salah (akurasi), jadi didominasi kelas Injury dan Malfunction. Untuk Death, lihat F1 dan recall per kelas.
''')
code(r'''
def pred_label(kunci):
    p = PROBA[kunci][1]; return p.argmax(1) if np.ndim(p) == 2 else np.asarray(p)
benar_j = pred_label(JUARA.kunci) == y_te
uji = []
for _, r in best.iterrows():
    if r.kunci == JUARA.kunci:
        continue
    benar_l = pred_label(r.kunci) == y_te
    b_, c_ = int((benar_j & ~benar_l).sum()), int((~benar_j & benar_l).sum())
    p = 1.0 if b_ + c_ == 0 else min(1.0, 2 * binom.cdf(min(b_, c_), b_ + c_, 0.5))
    uji.append({"pembanding": r.kunci, "juara_benar_lawan_salah": b_, "juara_salah_lawan_benar": c_, "p": p})
uji = pd.DataFrame(uji).sort_values("p").reset_index(drop=True)
if len(uji):
    m = len(uji); uji["p_holm"] = np.maximum.accumulate([min(1, p * (m - i)) for i, p in enumerate(uji.p)])
    uji["bermakna_5%"] = uji.p_holm < 0.05
uji.to_csv(TAB / "13_mcnemar.csv", index=False); display(uji.round(4))
''')
md("### 13.5 Kurva loss (model neural terbaik per keluarga)")
code(r'''
R = pd.DataFrame(RIWAYAT)
saring = [k for k in best.kunci if not k.startswith("ML")]
fig, axs = plt.subplots(1, max(1, len(saring)), figsize=(4.2 * max(1, len(saring)), 3.4), squeeze=False)
for ax, k in zip(axs[0], saring):
    kel, mod, imb, aug_m = k.split("|")
    imb_raw = {"ROS": "resample"}.get(imb, imb)
    awalan = {"DL": "BiLSTM", "DL hybrid": "Hybrid", "BART fine-tune": "BART", "DL hybrid + HP": "Hybrid+HP terbaik"}[kel]
    r = R[R.model.str.startswith(awalan) & R.model.str.endswith(f"|{imb_raw}|{aug_m}")]
    if kel == "DL hybrid":
        r = r[r.model == f"Hybrid|{imb_raw}|{aug_m}"]
    ax.plot(r.epoch, r.train_loss, "o-", label="train loss"); ax.plot(r.epoch, r.val_loss, "s-", label="val loss")
    ax2 = ax.twinx(); ax2.plot(r.epoch, r.val_macro_f1, "^--", c="g", label="val macro-F1"); ax2.set_ylim(0, 1)
    ax.set_title(f"{kel}\n{imb}, {aug_m}", fontsize=9); ax.set_xlabel("epoch"); ax.legend(fontsize=7, loc="upper right")
simpan_fig("13_kurva_loss")
''')

# =====================================================================================
md(r'''
## 14. Confusion matrix (data test)

Baris = label asli, kolom = prediksi. Angka atas = jumlah laporan, angka bawah = persentase dari baris (recall per kelas).
''')
code(r'''
def gambar_cm(ax, kunci, judul):
    cm = confusion_matrix(y_te, pred_label(kunci), labels=range(4)); cmn = cm / cm.sum(1, keepdims=True)
    annot = np.array([[f"{cm[i, j]}\n{cmn[i, j] * 100:.0f}%" for j in range(4)] for i in range(4)])
    sns.heatmap(cmn, annot=annot, fmt="", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax, xticklabels=KELAS, yticklabels=KELAS)
    ax.set_xlabel("prediksi"); ax.set_ylabel("label asli"); ax.set_title(judul, fontsize=9)
    return cm
n = len(best); kolom = 3; baris = math.ceil(n / kolom)
fig, axs = plt.subplots(baris, kolom, figsize=(5 * kolom, 4.3 * baris), squeeze=False)
for ax, (_, r) in zip(axs.ravel(), best.iterrows()):
    gambar_cm(ax, r.kunci, f"{r.keluarga}: {r.model[:32]}\n{r.imbalance}, {r.augmentasi} | macro-F1 {r.test_macro_f1:.3f}")
for ax in axs.ravel()[n:]:
    ax.axis("off")
simpan_fig("14_confusion_matrix_per_keluarga")
fig, ax = plt.subplots(figsize=(6, 5))
cm = gambar_cm(ax, JUARA.kunci, f"MODEL TERBAIK: {JUARA.model}\n{JUARA.imbalance}, {JUARA.augmentasi}")
pd.DataFrame(cm, index=KELAS, columns=KELAS).to_csv(TAB / "14_confusion_terbaik.csv")
simpan_fig("14_confusion_matrix_terbaik")
''')

# =====================================================================================
md(r'''
## 15. Analisis kesalahan (model terbaik, data test)

Setiap prediksi salah dibandingkan dengan label asli, lalu diberi **alasan** secara berurutan memakai aturan yang bisa diperiksa (aturan pertama yang cocok dipakai):

| Alasan | Aturan |
|---|---|
| Death tanpa kata kematian | label Death, narasi tidak memuat died/death/expired/passed away/deceased/fatal |
| Narasi menyebut kematian, kode pelapor bukan Death | prediksi Death, label lain, narasi memuat kata kematian |
| Other: gaya laporan lama | label Other, diterima ≤ 2010 |
| Diprediksi Other untuk laporan baru | prediksi Other, diterima > 2010 |
| Ada cedera/luka bakar tetapi dikode Malfunction | label Malfunction, prediksi Injury, narasi memuat kata cedera (burn, pain, blister, injur, hearing, ...) |
| Narasi template tanpa dampak pasien, dikode Injury | label Injury, prediksi Malfunction, narasi memuat "no patient / no injury / no adverse / could / potential" |
| Narasi pendek | < 30 kata |
| Kosakata tumpang tindih | selain di atas |

Prediksi salah dengan keyakinan ≥ 0,90 dicatat sebagai **kandidat salah label** yang perlu ditelaah manusia. Daftar ini pembacaan model, bukan kebenaran.
''')
code(r'''
P_te = PROBA[JUARA.kunci][1]
E = te[["mdr_report_key", "date_received", "label", "event_text", "n_kata", "tahun"]].copy()
E["prediksi"] = [KELAS[i] for i in pred_label(JUARA.kunci)]
E["keyakinan"] = P_te.max(1) if np.ndim(P_te) == 2 else np.nan
E["benar"] = E.label == E.prediksi
POLA_MATI = r"\b(died|dies|death|expired|passed away|deceased|fatal\w*|demise)\b"
POLA_CEDERA = r"\b(burn\w*|pain\w*|blister\w*|injur\w*|hearing|lacerat\w*|fractur\w*|wound\w*|treated|redness|erythema)\b"
POLA_TEMPLATE = r"\b(no patient|no injur\w*|no adverse|no harm|not involved|could|potential\w*)\b"
def alasan(r):
    t = r.event_text.lower()
    if r.label == "Death" and not re.search(POLA_MATI, t): return "Death tanpa kata kematian"
    if r.prediksi == "Death" and re.search(POLA_MATI, t): return "Narasi menyebut kematian, kode pelapor bukan Death"
    if r.label == "Other" and r.tahun <= 2010: return "Other: gaya laporan lama"
    if r.prediksi == "Other" and r.tahun > 2010: return "Diprediksi Other untuk laporan baru"
    if r.label == "Malfunction" and r.prediksi == "Injury" and re.search(POLA_CEDERA, t): return "Ada cedera/luka bakar, dikode Malfunction"
    if r.label == "Injury" and r.prediksi == "Malfunction" and re.search(POLA_TEMPLATE, t): return "Template tanpa dampak pasien, dikode Injury"
    if r.n_kata < 30: return "Narasi pendek (<30 kata)"
    return "Kosakata tumpang tindih"
S = E[~E.benar].copy(); S["alasan"] = S.apply(alasan, axis=1)
S["kandidat_salah_label"] = S.keyakinan >= 0.90
print(f"salah: {len(S)}/{len(E)} laporan test ({len(S) / len(E) * 100:.1f}%)")
pasangan = pd.crosstab(S.label, S.prediksi).reindex(index=KELAS, columns=KELAS, fill_value=0)
display(pasangan.rename_axis(index="label asli", columns="prediksi"))
''')
code(r'''
hit = S.alasan.value_counts()
fig, ax = plt.subplots(figsize=(9, 4))
b = ax.barh(hit.index[::-1], hit.values[::-1], color="#C44E52"); ax.bar_label(b, fontsize=8, padding=2)
ax.set_xlabel("jumlah prediksi salah"); ax.set_title(f"Alasan kesalahan model terbaik ({JUARA.model})")
simpan_fig("15_alasan_kesalahan")
ringkas = S.groupby(["label", "prediksi", "alasan"]).size().rename("jumlah").reset_index().sort_values("jumlah", ascending=False)
ringkas.to_csv(TAB / "15_ringkasan_kesalahan.csv", index=False); display(ringkas)
''')
md("### 15.1 Contoh kesalahan: label asli vs prediksi")
code(r'''
pd.set_option("display.max_colwidth", 220)
S["cuplikan"] = S.event_text.str.slice(0, 220)
kol_s = ["mdr_report_key", "tahun", "label", "prediksi", "keyakinan", "alasan", "cuplikan"]
S.sort_values(["alasan", "keyakinan"], ascending=[True, False]).drop(columns=["event_text"]).to_csv(TAB / "15_prediksi_salah.csv", index=False)
contoh = S.sort_values("keyakinan", ascending=False).groupby("alasan").head(2)
display(contoh[kol_s].round(3))
''')
md("### 15.2 Kesalahan yang dialami semua model terbaik (sulit untuk semua pendekatan)")
code(r'''
salah_semua = np.ones(len(te), bool)
for k in best.kunci:
    salah_semua &= pred_label(k) != y_te
X_sulit = E[salah_semua].copy(); X_sulit["alasan"] = X_sulit.apply(alasan, axis=1)
print(f"{salah_semua.sum()} laporan test salah diprediksi oleh SEMUA {len(best)} model terbaik")
display(X_sulit.groupby(["label", "alasan"]).size().rename("jumlah").reset_index().sort_values("jumlah", ascending=False))
''')
md("### 15.3 Kandidat salah label (prediksi salah dengan keyakinan ≥ 0,90)")
code(r'''
K = S[S.kandidat_salah_label].sort_values("keyakinan", ascending=False)
K[kol_s].to_csv(TAB / "15_kandidat_salah_label.csv", index=False)
print(f"{len(K)} kandidat (perlu ditelaah manusia; bukan kebenaran)"); display(K[kol_s].head(15).round(3))
''')

# =====================================================================================
md("## 16. Kesimpulan (dihitung otomatis dari run ini)")
code(r'''
imb_terbaik = pv_imb.mean().idxmax(); aug_terbaik = pv_aug.mean().idxmax()
alasan_top = hit.index[0] if len(hit) else "-"
teks = f"""
1. Data bersih {len(d)} laporan (train {len(tr)}, val {len(va)}, test {len(te)}); split per grup kemiripan tanpa grup menyeberang.
2. Model terbaik menurut val: {JUARA.model} [{JUARA.keluarga}] dengan {JUARA.imbalance} dan {JUARA.augmentasi}
   -> macro-F1 test {JUARA.test_macro_f1:.3f}, akurasi {JUARA.test_akurasi:.3f}, F1 Death {JUARA.test_f1_Death:.3f}, recall Death {JUARA.test_recall_Death:.3f}.
3. Terbaik per keluarga (macro-F1 test): """ + "; ".join(f"{r.keluarga} {r.test_macro_f1:.3f}" for _, r in best.iterrows()) + f"""
4. Penanganan ketidakseimbangan dengan rata-rata macro-F1 test tertinggi: {imb_terbaik}; augmentasi: {aug_terbaik}.
5. Uji McNemar-Holm: """ + ("; ".join(f"vs {r.pembanding.split('|')[0]} p_holm={r.p_holm:.3f}" for _, r in uji.iterrows()) if len(uji) else "-") + f"""
6. Alasan kesalahan terbanyak: {alasan_top}. {salah_semua.sum()} laporan test salah untuk semua model terbaik; {len(K)} kandidat salah label.
7. Test Death hanya {int((y_te == 2).sum())} laporan: satu laporan Death mengubah recall Death sebesar {100 / max(1, (y_te == 2).sum()):.0f} poin persen, jadi angka Death harus dibaca hati-hati.
"""
print(teks)
Path(TAB / "16_kesimpulan.txt").write_text(teks, encoding="utf-8")
json.dump({"waktu_detik": WAKTU, "device": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
           "smoke": SMOKE, "n": {"train": len(tr), "val": len(va), "test": len(te), "aug": len(aug)},
           "split_beku": beku,
           "versi": {m.__name__: m.__version__ for m in (np, pd, sklearn, scipy, torch, transformers)}},
          open(OUT / "run_info.json", "w"), indent=2)
print("waktu (menit):", {k: round(v / 60, 1) for k, v in WAKTU.items()})
''')

# =====================================================================================
nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(s) if t == M else nbf.v4.new_code_cell(s) for t, s in CELLS]
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}}
tujuan = ROOT / "notebooks" / "08_eksperimen_lengkap.ipynb"
nbf.write(nb, tujuan)
print("ditulis:", tujuan, "|", len(CELLS), "sel")
