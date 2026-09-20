# Stage 02: Comprehensive Exploratory Data Analysis (EDA) Engineering Report

**Document Version:** 1.0.0  
**Target Architecture Stage:** Stage 02 Deep Learning Multimodal Modeling  
**Prepared by:** Stage 02 EDA Engineer / Scientific AI Auditor  
**Date:** September 20, 2026  
**Status:** Completed, Audited & Scientifically Validated  

---

## 1. Executive Summary

This report establishes the baseline exploratory data analysis (EDA) for the **Stage 02 Deep Learning Oncology Dataset**, ensuring thorough empirical characterization of all three medical modalities prior to deep learning architecture design and training:
1. **Histopathology**: 2,500 curated 50x50 RGB patches for tissue and cancer pattern detection, data augmentation benchmarking, and convolutional neural network (CNN) classification.
2. **Radiology**: 800 axial CT slices paired 1:1 with 800 expert Gross Tumor Volume (GTV-1) segmentation masks across 7 non-small cell lung carcinoma (NSCLC) patients, enabling 2D/3D segmentation and volumetric estimation.
3. **Temporal Biomarkers**: 315 longitudinal plasma liquid biopsy observations across 94 prospective trial patients, tracking circulating tumor DNA (ctDNA) concentration, variant allele frequency (VAF), baseline PD-L1 expression, RECIST 1.1 treatment response, and overall survival outcome.

All analyses strictly preserve pre-existing patient-level and stratified split assignments (`train`, `val`, `test`), maintain zero patient cross-split leakage, and establish rigorous statistical foundations while adhering to data safety protocols.

---

## 2. Dataset Overview

| Metric / Dimension | Histopathology | Radiology (CT + GTV-1) | Temporal Biomarkers |
| :--- | :--- | :--- | :--- |
| **Primary Cohort Source** | Breast Histopathology IDC Benchmark | TCIA NSCLC-Radiomics (LUNG1) | INSPIRE Trial (Nature Cancer 2020) |
| **Total Samples / Records** | 2,500 image patches | 800 CT slices + 800 GTV masks | 315 serial liquid biopsy records |
| **Patient Count** | Anonymized multi-slide cohort | **Strictly 7 patients** | 94 patients |
| **Input Spatial / Temporal Dim** | 50 x 50 x 3 (RGB PNG) | 512 x 512 (8-bit Lung Window) | Serial observations (1 to 12 visits/pt) |
| **Primary Target Signal** | IDC vs Non-IDC / Benign | Binary GTV-1 Voxel Mask | ctDNA mutant molecules/mL, VAF, RECIST |
| **Split Architecture** | Stratified patch split (70/15/15) | Patient-level split (5 train / 1 val / 1 test) | Patient-level split (66 train / 14 val / 14 test) |

---

## 3. Histopathology EDA

### 3.1 Sample Count & Class Balance
- **Total Patches**: 2,500
- **Class 0 (`non_idc_or_benign`)**: 1,750 patches (**70.00%**)
- **Class 1 (`invasive_ductal_carcinoma`)**: 750 patches (**30.00%**)
- **Imbalance Ratio**: Exactly 2.33:1 (`non_idc_or_benign` to `invasive_ductal_carcinoma`), mirroring the natural incidence of malignant ductal infiltration in sampled whole-mount slide biopsies.

### 3.2 Partition Breakdown
Class proportions are strictly preserved across all splits:
- **Train (1,750 patches)**: 1,225 `non_idc_or_benign` (70.0%) | 525 `invasive_ductal_carcinoma` (30.0%)
- **Validation (375 patches)**: 262 `non_idc_or_benign` (69.87%) | 113 `invasive_ductal_carcinoma` (30.13%)
- **Test (375 patches)**: 263 `non_idc_or_benign` (70.13%) | 112 `invasive_ductal_carcinoma` (29.87%)

### 3.3 Image Integrity, Dimensions, and Deduplication
- **Spatial Resolution**: 100% of patches are verified at `(50, 50)` pixels.
- **Color Mode**: 100% standard 3-channel RGB.
- **Deduplication Audit**: Exactly 2,500 unique MD5 hashes. Zero duplicate images exist across or within partitions.

### 3.4 Pixel Intensity & Color Channel Statistics
Empirical channel-level distributions reveal marked morphological and histological distinctions:
- **Invasive Ductal Carcinoma (IDC)**:
  - Red Channel: Mean = 172.33 (Std = 20.84)
  - Green Channel: Mean = 137.97 (Std = 29.98)
  - Blue Channel: Mean = 186.27 (Std = 24.87)
  - Perceptual Luminance (Brightness): Mean = 153.73 (Std = 26.25)
  - Texture Contrast: Mean = 29.76 (Std = 10.79)
- **Non-IDC / Benign Tissue**:
  - Red Channel: Mean = 193.31 (Std = 24.12)
  - Green Channel: Mean = 171.77 (Std = 34.18)
  - Blue Channel: Mean = 216.39 (Std = 20.28)
  - Perceptual Luminance (Brightness): Mean = 183.28 (Std = 28.98)
  - Texture Contrast: Mean = 33.43 (Std = 12.78)

**Histopathological Interpretation**:
IDC patches exhibit substantially darker overall luminance (153.73 vs 183.28) and elevated hematoxylin staining (absorption in Green channel yielding lower mean G values: 137.97 vs 171.77). This corresponds to dense hyperchromatic malignant cell nuclei, high nuclear-to-cytoplasmic (N:C) ratios, and structural desmoplasia displacing hypocellular adipose and fibrous background stroma.

### 3.5 Generated Histopathology Visualizations
1. `01_histopathology_class_and_split_dist.png`: Bar plots showing overall 70/30 class balance and consistency across train, val, and test partitions.
2. `02_histopathology_rgb_channel_distributions.png`: Kernel density estimates (KDE) comparing Red, Green, and Blue intensity curves by pathology class.
3. `03_histopathology_brightness_contrast.png`: Scatter plot and violin plots illustrating luminance vs contrast separation between benign and malignant tissue.
4. `04_histopathology_representative_patches.png`: 4x6 sample montage displaying representative 50x50 patches from each split and class.

---

## 4. Radiology EDA

### 4.1 Cohort Architecture & The Patient-vs-Slice Distinction
> [!IMPORTANT]
> **CRITICAL SCIENTIFIC DISTINCTION**:
> The radiology dataset comprises **800 axial CT slices** originating from **strictly 7 unique patients** (LUNG1-001 through LUNG1-007).
> Slices represent repeated z-axis volumetric slices along the patient's thorax.
> **DO NOT** calculate or present patient-level statistics as if each slice were an independent patient, and **DO NOT** imply that 800 slices represent 800 patients.

### 4.2 Patient-Level Summary

| Patient ID | Split | Total Slices | GTV Tumor Slices | Tumor Ratio (%) | 3D GTV Volume (cm³) | Stage | Histology | 2-Yr Mortality |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LUNG1-001** | Train | 134 | 21 | 15.67% | 156.32 | IIIb | Large cell | 0 (Surviving) |
| **LUNG1-002** | Train | 111 | 26 | 23.42% | 359.47 | I | Squamous cell | 1 (Deceased) |
| **LUNG1-003** | Train | 107 | 17 | 15.89% | 34.82 | IIIb | Large cell | 1 (Deceased) |
| **LUNG1-004** | Train | 114 | 36 | 31.58% | 84.46 | II | Squamous cell | 1 (Deceased) |
| **LUNG1-005** | Train | 91 | 24 | 26.37% | 83.55 | IIIb | Squamous cell | 1 (Deceased) |
| **LUNG1-006** | Val | 114 | 23 | 20.18% | 78.34 | IIIa | Squamous cell | 1 (Deceased) |
| **LUNG1-007** | Test | 129 | 10 | 7.75% | 12.49 | IIIa | Squamous cell | 1 (Deceased) |

### 4.3 Slices and Patient Split Distribution
- **Train Partition**: 5 patients (71.4% of patients), **557 slices** (124 tumor slices, 433 non-tumor slices).
- **Validation Partition**: 1 patient (`LUNG1-006`, 14.3% of patients), **114 slices** (23 tumor slices, 91 non-tumor slices).
- **Test Partition**: 1 patient (`LUNG1-007`, 14.3% of patients), **129 slices** (10 tumor slices, 119 non-tumor slices).
- **Patient Leakage**: **0.0%**. No patient spans multiple partitions.

### 4.4 Tumor Presence and Segmentation Mask Coverage
- **Total Slices**: 800
- **Tumor-Positive Slices (GTV-1 Present)**: 157 (**19.62%**)
- **Tumor-Negative Slices (Background Lung/Thorax)**: 643 (**80.38%**)
- **Tumor Cross-Sectional Area (on 157 positive slices)**:
  - Mean: 1,718.59 mm²
  - Standard Deviation: 1,701.47 mm²
  - Median: 1,018.48 mm²
  - Interquartile Range (25% - 75%): 594.67 mm² - 1,997.83 mm²
  - Range: 43.87 mm² (apical tumor margin) to 6,693.16 mm² (bulky primary central tumor in LUNG1-002)
- **Mask Coverage Percentage**:
  - Out of 512x512 pixels (262,144 total pixels per slice), positive tumor masks occupy:
    - Mean: **0.69%** of slice area
    - Median: **0.41%**
    - Range: **0.02% to 2.67%**
  - **Sparsity Finding**: The segmentation signal is highly sparse on the full 512x512 canvas. Deep learning architectures (e.g. U-Net) must address extreme foreground/background pixel imbalance.

### 4.5 Generated Radiology Visualizations
1. `01_radiology_patient_and_split_breakdown.png`: Stacked bar chart showing total slices and tumor slice counts per patient, alongside patient/slice split distributions.
2. `02_radiology_tumor_area_distributions.png`: Histogram and patient-level boxplots of 2D tumor cross-sectional area (mm²).
3. `03_radiology_gtv_volumes_and_coverage.png`: Patient 3D Gross Tumor Volume (cm³) grouped by clinical stage and histogram of slice mask coverage percentages.
4. `04_radiology_representative_ct_and_masks.png`: High-resolution comparison of axial CT lung window slices and paired red GTV-1 segmentation contour overlays.

---

## 5. Biomarker EDA

### 5.1 Cohort Dimensions & Visit Frequency
- **Total Longitudinal Observations**: 315 records
- **Unique Patients**: 94 patients
- **Serial Records per Patient**:
  - Minimum: 1 visit
  - 25th Percentile: 1 visit
  - Median: 2.0 visits (Mean: 3.35 visits)
  - 75th Percentile: 4 visits
  - Maximum: 12 visits
  - Patients with serial follow-up (>=2 visits): 74 patients (78.7%)

### 5.2 Chronological Validity & Temporal Distribution
- **Chronological Violation Check**: **0 violations**. All patient records are monotonically non-decreasing in `days_from_baseline`.
- **Duplicate Visit Check**: **0 duplicate patient-timepoint records**.
- **Temporal Horizon Distribution**:
  - Baseline records (`days_from_baseline <= 0`): 92 records (across 92 patients with verified baseline assays; 2 baseline records have missing offset days).
  - Post-baseline longitudinal observations: 221 records spanning up to 799 days post-treatment initiation.
  - Cycle 3 landmark (~6 weeks / Day 35 - 65): 71 patients have high-powered evaluations at this critical therapeutic window.

### 5.3 Biomarker Distributions & Quantile Statistics

| Statistic | ctDNA Concentration (molecules/mL) | Variant Allele Frequency (VAF fraction) | Baseline PD-L1 Expression (MPS %) |
| :--- | :--- | :--- | :--- |
| **Count** | 315 valid | 315 valid | 312 valid (3 missing) |
| **Mean** | 342.76 | 0.0703 (7.03%) | 23.54% |
| **Std Dev** | 1,110.81 | 0.1226 (12.26%) | 37.89% |
| **Min** | 0.00 | 0.0000 (0.00%) | 0.00% |
| **25%** | 0.00 | 0.0000 (0.00%) | 0.00% |
| **50% (Median)** | **9.42** | **0.0089 (0.89%)** | **1.00%** |
| **75%** | 171.09 | 0.0868 (8.68%) | 20.00% |
| **Max** | 12,916.44 | 0.7597 (75.97%) | 100.00% |

**Clinical Distribution Observations**:
- **Severe Right-Skewness**: ctDNA mutant molecules span over 4 orders of magnitude (0 to 12,916 molecules/mL). 84 records (26.7%) have zero detectable ctDNA (`ctdna_detected == 0`). A logarithmic transformation, e.g. $\log_{10}(\text{ctDNA} + 1)$, is mathematically essential.
- **PD-L1 Bimodality**: Baseline PD-L1 expression is heavily clustered at 0% (median = 1.0%), with a secondary tail reaching 100%, reflecting standard clinical tumor proportion score (TPS) cutoff behaviors (<1%, 1-49%, >=50%).

### 5.4 Clinical Response and Survival Outcomes
- **RECIST 1.1 Best Overall Response** (315 observations):
  - Partial Response (PR): 118 (37.5%)
  - Progressive Disease (PD): 90 (28.6%)
  - Stable Disease (SD): 76 (24.1%)
  - Complete Response (CR): 27 (8.6%)
  - Not Evaluable (NE) / Unknown: 4 (1.3%)
- **Patient Long-Term Vital Status**:
  - Living: 173 observations
  - Deceased: 140 observations
  - Unknown: 2 observations

### 5.5 Generated Biomarker Visualizations
1. `01_biomarkers_longitudinal_timelines.png`: Distribution of visits per patient and swimmer timeline plot tracking serial liquid biopsy dates for the top 25 followed patients colored by RECIST response.
2. `02_biomarkers_ctdna_and_vaf_distributions.png`: Log-scale histogram of ctDNA mutant molecules/mL and percentage distribution of VAF.
3. `03_biomarkers_pdl1_response_outcome.png`: Baseline PD-L1 expression distribution and stacked bar chart showing RECIST response categories cross-tabulated with overall survival.
4. `04_biomarkers_serial_trajectories.png`: Spaghetti trajectory plots showing longitudinal $\log_{10}(\text{ctDNA} + 1)$ dynamic changes over time categorized by Partial Response, Stable Disease, and Progressive Disease.

---

## 6. Split Analysis & Leakage Verification

EDA confirms complete isolation across partitions for all three modalities:
1. **Histopathology**: Partitioned with stratified balancing (70% Class 0 / 30% Class 1 in each partition).
2. **Radiology**: Strict patient-level grouping. Training set (LUNG1-001 through 005), Validation set (LUNG1-006), Test set (LUNG1-007). Zero slice overlap.
3. **Biomarkers**: Strict patient-level grouping. 66 patients (199 observations) in Train, 14 patients (50 observations) in Validation, 14 patients (66 observations) in Test.

Zero test-set information was utilized for parameter fitting, tuning, or threshold selection.

---

## 7. Missing Data Analysis

| Field | Modality | Missing Count | % Missing | Mechanistic Explanation & Handling Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| `protein_marker` / `protein_marker_name` | Biomarkers | 3 records | 0.95% | 1 patient lacked baseline biopsy tissue for PD-L1 IHC. Impute via median or indicator flag. |
| `days_from_baseline` / `weeks_from_baseline` | Biomarkers | 2 records | 0.63% | Date of baseline biopsy not recorded; exclude or impute relative to cycle start. |
| `os_months` | Biomarkers | 2 records | 0.63% | Patient lost to follow-up before survival census. |
| `tumor_volume` | Biomarkers | 315 records | 100.0% | **Clinically expected**: Liquid biopsy measures circulating cell-free DNA fragments in plasma; anatomical caliper volume is not recorded in liquid biopsy assays. |
| `vaf_change_percent` / `molec_change_percent` | Biomarkers | 10 records | 3.17% | Baseline records (which have no prior visit to compute relative change against) or division by zero baseline ctDNA. Handled via clipping or log-difference. |

---

## 8. Data Quality Observations

1. **Pixel Integrity**: 100% of histopathology patches (2,500) and CT slices/masks (800) are valid, uncorrupted, and conform to exact expected spatial dimensions (50x50 and 512x512).
2. **Mask Pairing**: 100% 1:1 pairing between CT scans and binary segmentation masks. Zero orphaned masks.
3. **Deduplication**: 0 duplicate image hashes in histopathology, 0 duplicate visits in longitudinal biomarkers.
4. **Chronological Validity**: Zero chronological sequencing errors in serial biomarker timestamps.

---

## 9. Cohort / Linkage Limitations

> [!WARNING]
> **MULTI-COHORT INDEPENDENCE**:
> - The **Radiology cohort** consists of 7 patients from the TCIA NSCLC-Radiomics study (lung adenocarcinoma / squamous cell carcinoma).
> - The **Biomarker cohort** consists of 94 patients from the prospective INSPIRE immunotherapy trial.
> - The **Histopathology cohort** consists of 2,500 patches from a breast invasive ductal carcinoma (IDC) series.
> - **These cohorts are completely distinct and originate from different medical centers, patient populations, and clinical trials.**
> - **There is NO shared patient identifier or same-patient longitudinal linkage established between radiology and biomarkers.**
> - Multimodal deep learning architectures must model these modalities as specialized domain-specific encoders (e.g. modular vision encoders, sequence encoders) rather than assuming single-patient end-to-end linked tuples.

---

## 10. Deep Learning Implications

The empirical findings from this EDA establish the following critical guidelines for downstream deep learning modeling:

1. **Histopathology Modeling**:
   - **Class Imbalance**: The 70/30 class imbalance warrants weighted cross-entropy loss ($\text{weight}_{\text{pos}} = 2.33$) or Focal Loss ($\gamma = 2.0, \alpha = 0.30$) to optimize the minority IDC detection.
   - **Color Augmentation**: Given the wide spread in hematoxylin/eosin staining luminance (mean 153.7 IDC vs 183.3 Benign), models require stain normalization (e.g., Macenko / Vahadane method) or vigorous color jitter (brightness, contrast, saturation) alongside random affine rotations and flips.
2. **Radiology Segmentation**:
   - **Extreme Pixel Sparsity**: With foreground tumor masks occupying only 0.69% of the 512x512 canvas, standard Binary Cross-Entropy will lead to background collapse. Downstream models (U-Net, Attention U-Net) must utilize a hybrid Dice Loss + Focal Loss formulation.
   - **Small Patient Cohort**: Because the 800 slices stem from only 7 patients, cross-slice correlation within the same patient is high. Models evaluated on `val` (LUNG1-006) and `test` (LUNG1-007) test out-of-patient generalizability. Patch-based extraction or heavy spatial data augmentation is essential to avoid overfitting to patient-specific thoracic anatomies.
3. **Longitudinal Biomarker Sequence Modeling**:
   - **Extreme Skewness**: Due to 4 orders of magnitude in ctDNA mutant molecules/mL, input features must be normalized using $\log_{10}(\text{ctDNA} + 1)$ and robust scaling.
   - **Irregular Time Steps**: Longitudinal visits occur at irregular intervals (e.g., Day -7, Day 43, Day 111). Recurrent architectures (LSTM, GRU) or Transformers should incorporate continuous time embeddings (e.g., $\Delta t$ encoding) rather than assuming equidistant discrete steps.
   - **Landmark Target**: Predicting 6-week Cycle 3 molecular response (clearance vs persistence) provides high statistical power (71 evaluable patients) compared to arbitrary fixed-day horizons.

---

## 11. Recommended Next Steps

1. **Feature / Encoder Engineering**:
   - Develop a lightweight CNN / ResNet backbone for 50x50 IDC patch classification with stain augmentation.
   - Develop a 2D U-Net / Residual U-Net segmentation pipeline for CT slices with Dice loss.
   - Develop an LSTM / GRU / MLP trajectory forecasting model for longitudinal ctDNA dynamics.
2. **Cross-Validation Framework**:
   - For radiology, implement Leave-One-Patient-Out or grouped k-fold cross-validation when doing hyperparameter ablation on the training partition.
3. **Multimodal Fusion Strategy**:
   - Structure multimodal integration at the late-fusion / decision-engine level, combining risk logits from specialized modular heads rather than forcing early tensor concatenation across unlinked cohorts.
