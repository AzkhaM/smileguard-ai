# 🦷 SmileGuard AI

**Multimodal Dental Screening untuk Klinik: TMJ Risk Indicator + Caries Detection**

Self-project AI screening tool untuk klinik gigi paman. Pasien upload foto intraoral
+ jawab kuesioner gejala → AI berikan indikasi awal (bukan diagnosis) → dokter di klinik
melakukan verifikasi & treatment plan.

> ⚠️ **Disclaimer**: Output AI adalah *screening / risk indicator*, BUKAN diagnosa medis.
> Semua hasil harus diverifikasi oleh dokter berlisensi.

---

## 🎯 Goals

- **Primary**: TMJ risk screening (multimodal: foto intraoral + kuesioner gejala)
- **Secondary**: Caries detection & severity classification
- **Bonus**: LLM explainer narasi Bahasa Indonesia ramah pasien

## 🏗️ Architecture

```
[App] → [Quality Gate] → [YOLO Caries] + [TMJ Indicators] → [Fusion Model] → [LLM Explainer] → [Doctor Dashboard]
```

5-stage pipeline. Detail di docs/architecture.md (TBD).

---

## 📁 Project Structure

```
smileguard/
├── README.md                    # this file
├── data/
│   ├── raw/                     # AKU dataset, AlphaDent (gitignored)
│   ├── processed/               # train/val/test splits, YOLO format
│   └── annotations/             # custom annotations dari klinik paman
├── notebooks/
│   ├── 00_eda_aku_dataset.ipynb       # ← Phase 0 mulai dari sini
│   ├── 01_train_baseline_yolo.ipynb   # ← Phase 1
│   ├── 02_hyperparam_tuning.ipynb
│   ├── 03_tmj_indicators.ipynb
│   └── 04_multimodal_fusion.ipynb
├── src/
│   ├── data/                    # data loaders, preprocessing
│   ├── models/                  # model definitions
│   ├── training/                # training scripts
│   ├── inference/               # inference pipeline
│   └── api/                     # FastAPI serving (Phase 4)
├── configs/                     # YOLO config, training params
├── experiments/                 # WandB run logs, checkpoints
└── docs/                        # architecture, API spec, dataset notes
```

---

## 📅 Roadmap (8 weeks)

| Week | Phase | Goal |
|------|-------|------|
| 1 | Foundation | Setup, dataset request, EDA |
| 2 | Caries Baseline | YOLOv11s mAP > 0.80 |
| 3 | Caries Iteration | S1-S5 scenarios, best model |
| 4 | TMJ Indicators | Rule-based + tooth wear detector |
| 5 | Questionnaire | 15 soal valid + tabular features |
| 6 | Multimodal Fusion | XGBoost/MLP fusion |
| 7 | LLM + API | FastAPI + Claude/Gemini explainer |
| 8 | Demo Deploy | HuggingFace Spaces + integration test |

---

## 🗃️ Datasets

### Tier 1 — Caries (siap pakai)
- **AKU Intraoral Dataset** (Aga Khan University, 2025)
  - 6,313 images, YOLO/COCO/VOC format
  - Zenodo: https://zenodo.org/records/14769743
  - DOI: 10.5281/zenodo.14827784
  - ⚠️ Status: Image Restricted — request access via Zenodo (justifikasi: AI healthcare project)

- **AlphaDent** (Sechenov University, 2025)
  - GitHub: https://github.com/AlphaChip-LLC/AlphaDent (cek availability)
  - Paper: arXiv 2507.22512
  - 6 caries classes, intraoral DSLR photos

- **Roboflow Universe — Dental Caries**
  - https://universe.roboflow.com (search: dental caries)
  - Multiple datasets, free with account

### Tier 2 — TMJ Indicators (kombinasi)
- Tooth wear datasets (Kaggle search: "tooth wear")
- Orthodontic image datasets (Roboflow Universe)
- Custom: re-annotate subset AKU dataset untuk attrition/maloklusi

### Tier 3 — Local (jangka panjang)
- Foto dari klinik paman (consent dari pasien)
- Anotasi oleh paman (dokter berlisensi)

### Reference
- **Dental Datasets Compilation**: https://github.com/sergiouribe/dental_datasets_itu

---

## ⚙️ Tech Stack

- **Python** 3.10+
- **Ultralytics YOLOv11** (object detection)
- **OpenCV + MediaPipe** (preprocessing, quality gate)
- **scikit-learn / XGBoost** (multimodal fusion)
- **Weights & Biases** (experiment tracking, free tier)
- **FastAPI + ONNX Runtime** (serving — Phase 4)
- **Claude API / Gemini Flash** (LLM explainer)

## 💻 Compute

- Local laptop: dev, debugging, EDA
- Kaggle Notebooks (P100, 30hr/week): main training
- Google Colab Free (T4): light experiments

---

## 🚀 Quick Start

### Setup environment
```bash
# Conda recommended
conda create -n smileguard python=3.10 -y
conda activate smileguard

# Core deps
pip install ultralytics opencv-python pillow pandas numpy matplotlib seaborn
pip install scikit-learn xgboost
pip install jupyter ipywidgets
pip install wandb  # optional but recommended
```

### Step 1: Request AKU dataset
1. Visit https://zenodo.org/records/14769743
2. Click "Request access"
3. Justifikasi: "Academic AI healthcare screening project, recent Informatics graduate, building dental screening tool with proper medical disclaimer & doctor verification loop"
4. Tunggu approval (biasanya 1-7 hari)

### Step 2: Setup Kaggle/Colab
- Buat akun Kaggle, verify phone untuk akses GPU
- Pin notebook `01_train_baseline_yolo.ipynb` ke Kaggle

### Step 3: Run EDA
- Buka `notebooks/00_eda_aku_dataset.ipynb`
- Jalankan cell by cell
- Pahami distribusi data sebelum training

---

## 📖 References

- Ahmed et al. (2025). "Annotated intraoral image dataset for dental caries detection." *Scientific Data* 12:1297.
- AlphaDent paper (2025). arXiv 2507.22512.
- Choi et al. (2024). "Deep learning for TMJ disorder detection from panoramic radiographs."

---

## 📝 Ethical & Legal Notes

- **Output positioning**: "Screening / Risk Indicator" — tidak boleh diklaim sebagai "diagnosis"
- **Disclaimer wajib** di setiap output ke pasien
- **Doctor-in-the-loop**: hasil AI selalu di-review dokter sebelum dikirim ke pasien
- **Data privacy**: foto pasien harus encrypted, consent eksplisit, complies UU PDP Indonesia
- **Klaim marketing**: hindari "AI dokter gigi" — gunakan "AI screening assistant"

---

**Maintainer**: Riexu (Azkha Mardiyan Muttaqien)
**Status**: 🚧 Phase 0 — Foundation
**License**: TBD (private project)
