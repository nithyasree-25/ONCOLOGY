# Stage 02: Deep Learning Multimodal Dataset

## 1. Purpose of Stage 02 Dataset
This dataset provides an engineered, standardized, audited, and validated multimodal foundation for developing and evaluating deep learning architectures in oncology:
1. **Histopathology Computer Vision**: 2,500 curated 50x50 RGB patches for tissue/cancer pattern detection, convolutional neural networks (CNNs), data augmentation pipelines, and Grad-CAM interpretability.
2. **Radiological Imaging**: 800 axial CT slices paired with 800 expert Gross Tumor Volume (GTV-1) segmentation masks from 7 NSCLC patients in TCIA, supporting 2D/3D tumor localization, U-Net segmentation, slice tumor area analysis, and total 3D tumor volume computation.
3. **Temporal Biomarkers**: 315 serial plasma circulating tumor DNA (ctDNA) observations across 94 patients from the prospective INSPIRE trial, supporting longitudinal sequence modeling (LSTM, GRU, Transformers) of tumor burden, mutant molecules/mL, VAF %, baseline PD-L1 expression, RECIST 1.1 response, and overall survival.
4. **Multimodal Integration**: Providing cross-modality complementary inputs for downstream multimodal fusion architectures.

---

## 2. Directory Structure

`
Stage02_Data/
|
|-- histopathology/
|   |-- images/              # 2,500 50x50 RGB PNG patches
|   +-- labels.csv           # Patch metadata, cancer binary label, class, split, md5
|
|-- radiology/
|   |-- images/              # 800 512x512 8-bit windowed CT slices (PNG)
|   |-- masks/               # 800 512x512 binary GTV-1 segmentation masks (PNG)
|   +-- metadata.csv         # Patient ID, slice ID, tumor area (mm^2), tumor volume (cm^3), TNM stage, survival
|
|-- biomarkers/
|   +-- longitudinal.csv     # 315 serial ctDNA & PD-L1 observations across 94 patients
|
|-- metadata/
|   |-- dataset_sources.csv  # Official URLs, licenses, citations, sample sizes
|   |-- data_dictionary.csv  # Schema documentation for all 53 columns
|   |-- split_manifest.csv   # Unified ML split manifest (train, val, test)
|   +-- data_quality_report.md # Comprehensive cleaning and audit report
|
+-- README.md
`

---

## 3. Dataset Modalities & Official Sources

| Modality | Primary Dataset Name | Official Source / Repository | License | Patient Count | Sample Count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Histopathology** | Breast Histopathology Images (IDC Benchmark) | Hugging Face (dbzadnen/breast-histopathology-images) / Cruz-Roa et al. | CC0-1.0 (Public Domain) | 162 WSI cohort | 2,500 patches |
| **Radiology** | TCIA NSCLC-Radiomics (LUNG1 Cohort) | The Cancer Imaging Archive (TCIA) / Aerts et al. | CC BY 3.0 | 7 patients | 800 CT slices + 800 GTV masks |
| **Biomarkers** | INSPIRE Prospective ctDNA Trial Cohort | Pugh Lab INSPIRE ctDNA / Bratman et al. Nature Cancer 2020 | MIT / Open Academic | 94 patients | 315 longitudinal visits |

---

## 4. Preprocessing & Data Engineering Workflow

### 4.1 Histopathology
- **Filtering**: Candidate patches from whole-slide acquisitions were verified with PIL. 109 edge-truncated patches of irregular dimensions were removed.
- **Integrity**: MD5 hashes computed to verify 0 duplicate images.
- **Standardization**: All images are standard 50x50x3 8-bit RGB PNG format.
- **Labels**: Binary indicator 0 (non_idc_or_benign) and 1 (invasive_ductal_carcinoma), maintaining a natural 70% / 30% class distribution.

### 4.2 Radiology
- **Conversion & Windowing**: High-bit depth CT volumes in Hounsfield Units (HU) converted from volumetric NIfTI to 2D axial PNG slices using clinical lung windowing (Center = -600 HU, Width = 1500 HU, Range = [-1350, +150 HU]) scaled linearly to [0, 255] uint8.
- **Paired Segmentation**: Expert Gross Tumor Volume (GTV-1) delineations thresholded at >0, yielding 800 paired binary masks (512x512).
- **Physical Metrics**: Voxel spacing extracted to compute slice tumor cross-sectional area (mm^2) and patient 3D tumor volume (cm^3 and mm^3).
- **Clinical Integration**: Matched with patient AJCC clinical stage, tumor histology, 2-year mortality outcome, and overall survival time.

### 4.3 Longitudinal Biomarkers
- **Standardization**: Mapped serial plasma liquid biopsy assays into a standardized temporal schema.
- **Metrics**: Quantitative ctDNA mutant molecules per mL (ctdna_value), mean variant allele frequency (vaf), baseline PD-L1 expression percentage (protein_marker), RECIST 1.1 treatment response, and overall survival outcome.
- **Chronological Sorting**: Every sequence is strictly ordered by days_from_baseline.

---

## 5. Train / Validation / Test Splits & Data Leakage Prevention

- **Patient-Level Splitting**: All radiology and biomarker splits are partitioned at the **patient identifier level**:
  - **Radiology (7 Patients)**:
    - train: LUNG1-001, LUNG1-002, LUNG1-003, LUNG1-004, LUNG1-005 (557 slices, 71.4% of patients)
    - val: LUNG1-006 (114 slices, 14.3% of patients)
    - test: LUNG1-007 (129 slices, 14.3% of patients)
  - **Biomarkers (94 Patients)**:
    - train: 66 patients (199 serial observations, ~70%)
    - val: 14 patients (50 serial observations, ~15%)
    - test: 14 patients (66 serial observations, ~15%)
  - **Histopathology (2,500 Patches)**:
    - Stratified patch-level split: 1,750 train (70%) / 375 val (15%) / 375 test (15%).
- Complete split assignments are unified in metadata/split_manifest.csv.

---

## 6. How the EDA Engineer Should Load the Data

All file paths are relative to Stage02_Data/.

`python
import os
import pandas as pd
from PIL import Image

BASE_DIR = 'Stage02_Data'

# 1. Load Histopathology
df_histo = pd.read_csv(os.path.join(BASE_DIR, 'histopathology', 'labels.csv'))
sample_img = Image.open(os.path.join(BASE_DIR, df_histo.iloc[0]['image_path']))
print('Histopathology:', df_histo.shape, 'Sample format:', sample_img.size, sample_img.mode)

# 2. Load Radiology
df_rad = pd.read_csv(os.path.join(BASE_DIR, 'radiology', 'metadata.csv'))
sample_ct = Image.open(os.path.join(BASE_DIR, df_rad.iloc[0]['image_path']))
sample_mask = Image.open(os.path.join(BASE_DIR, df_rad.iloc[0]['mask_path']))
print('Radiology:', df_rad.shape, 'CT format:', sample_ct.size, 'Mask format:', sample_mask.size)

# 3. Load Longitudinal Biomarkers
df_bio = pd.read_csv(os.path.join(BASE_DIR, 'biomarkers', 'longitudinal.csv'))
print('Biomarkers:', df_bio.shape, 'Unique patients:', df_bio['patient_id'].nunique())

# 4. Load Split Manifest & Data Dictionary
df_splits = pd.read_csv(os.path.join(BASE_DIR, 'metadata', 'split_manifest.csv'))
df_dict = pd.read_csv(os.path.join(BASE_DIR, 'metadata', 'data_dictionary.csv'))
`

---

## 7. Scientific Limitations

1. **Pathology Scope**: The histopathology dataset is a breast invasive ductal carcinoma (IDC) primary tissue patch dataset. It is NOT a lymph node micro-metastasis dataset (such as CAMELYON16) nor an explicit surgical margin margin-ink dataset. It serves as a pathology classification proxy for tissue pattern detection, CNN classification, and Grad-CAM saliency.
2. **Pathology Patient Leakage Status**: Original patch filenames in the upstream parquet were anonymized sequentially without preserving patient/slide IDs. Therefore, patient-level leakage cannot be independently audited from the retained patches.
3. **Radiology Cohort Size**: The CT cohort comprises 7 patients (800 slices). Slices represent repeated axial measurements along the z-axis of these 7 subjects. It is suitable for 2D/3D segmentation and volumetric experimentation, but NOT for population-level clinical generalization.
4. **Independent Unlinked Cohorts**: The INSPIRE biomarker cohort (94 patients) and LUNG1 CT cohort (7 patients) are separate clinical cohorts and are NOT patient-linked.
5. **3-Month Forecasting Target**: A strict 90-day timepoint is sparse (29 patients have observations in the 60-120 day window). The recommended and scientifically robust temporal target is Cycle 3 landmark prediction (35-65 days, available for 71 patients), or 6-month progression/overall survival modeling.

**Dataset Version**: 2.1.0  
**Release Date**: September 20, 2026