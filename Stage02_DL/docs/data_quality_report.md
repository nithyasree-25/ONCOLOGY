# Stage 02 Data Quality & Cleaning Report

**Prepared by:** Stage 02 Data Engineer / Scientific Auditor  
**Date:** September 20, 2026  
**Status:** Completed, Audited & Validated  

---

## 1. Executive Summary

This report documents the end-to-end data acquisition, filtering, deduplication, integrity auditing, and cleaning operations executed for the Stage 02 (Deep Learning) multi-modal cancer dataset.

The dataset encompasses three complementary modalities engineered from reputable, public, and scientifically validated cohorts:
1. **Histopathology**: 2,500 curated 50x50 RGB PNG patches for cancer pattern detection, CNN classification, and Grad-CAM interpretability.
2. **Radiology**: 800 axial CT slices with 800 paired binary Gross Tumor Volume (GTV-1) segmentation masks across 7 NSCLC patients from TCIA, with exact slice tumor area (mm^2) and total 3D tumor volume (cm^3 and mm^3).
3. **Temporal Biomarkers**: 315 longitudinal plasma ctDNA observations across 94 cancer patients from the prospective INSPIRE trial, tracking mutant molecules/mL, mean VAF %, baseline PD-L1 expression, RECIST 1.1 response, and overall survival outcome.

---

## 2. Modality Cleaning & Audit Details

### 2.1 Histopathology Data Cleaning & Provenance Audit

- **Original Candidate Pool**: 13,850 patches extracted from the standard Invasive Ductal Carcinoma (IDC) Whole Mount Slide benchmark (val.parquet).
- **Exact Provenance**: Cruz-Roa et al. (2014) / Janowczyk et al. (2016) breast histopathology whole-slide image series (162 patients, 40x magnification).
- **Corruption & Dimension Verification**:
  - All 13,850 image byte streams were opened and inspected via PIL.
  - **109 edge patches** were detected with non-standard dimensions (e.g., 50x38, 50x28, 50x6 pixels) resulting from whole-slide tissue boundary truncations in the raw acquisition grid.
  - **Reason for Removal**: Non-standard aspect ratios distort spatial CNN feature maps and receptive fields unless zero-padded or stretched.
  - **Action Taken**: Filtered out all edge-truncated patches to retain strictly 50x50x3 RGB uncorrupted images (13,741 strictly valid patches).
- **Label Nomenclature & Clinical Accuracy**:
  - In original Whole Mount Slide IDC datasets, label 0 corresponds to tissue regions outside IDC annotations (which includes normal glandular breast parenchyma, fibrous stroma, benign hyperplasia, or fibroadenomatous elements).
  - Label 0 is therefore strictly designated as 
on_idc_or_benign rather than pure uniform normal stroma.
  - Label 1 corresponds to confirmed invasive_ductal_carcinoma.
- **Deduplication Check**:
  - MD5 hashes were computed across all candidates: 0 duplicate image hashes.
- **Patient/Slide Identifiers & Leakage Status**:
  - The upstream parquet anonymized patch filenames sequentially (e.g., batch_100_sample_*.png).
  - As slide and patient IDs were not embedded in the parquet metadata, patient-level leakage cannot be independently audited from the retained patches.
- **Stratified Subsetting & Balancing**:
  - Sampled exactly **2,500 patches** preserving the natural clinical positive rate:
    - **Class 0 (non_idc_or_benign)**: 1,750 patches (70.0%)
    - **Class 1 (invasive_ductal_carcinoma)**: 750 patches (30.0%)
- **Data Partitioning**:
  - Split: **1,750 train (70%) / 375 val (15%) / 375 test (15%)**, maintaining identical class proportions across splits.

---

### 2.2 Radiology Data Cleaning & Standardization

- **Original Candidate Pool**: 422 volumetric CT scans with RTSTRUCT / NIfTI segmentation from the TCIA NSCLC-Radiomics collection.
- **Target Selection**:
  - Selected 7 complete consecutive patients (LUNG1-001 through LUNG1-007) representing Stage I to Stage IIIb non-small cell lung carcinoma with expert GTV-1 (Gross Tumor Volume) segmentations.
- **DICOM / NIfTI Conversion & Windowing**:
  - Raw CT Hounsfield Units (HU) span [-1024, +3071]. Standard lung parenchyma windowing was applied:
    - Center (Window Level): -600 HU
    - Width: 1500 HU
    - Clipping range: [-1350, +150 HU]
    - Rescaled linearly to 8-bit unsigned integer range [0, 255] for standard vision network ingestion.
- **Mask Integrity & Alignment**:
  - Binary GTV-1 masks were extracted, thresholded at >0, and verified to match CT slices slice-for-slice:
    - 512x512 pixel dimensions.
    - Zero orphaned masks: Every slice has an exact corresponding mask file (800 CT images, 800 mask images).
    - 157 slices contain positive tumor delineations; 643 slices contain surrounding lung / mediastinal tissue.
- **Tumor Metric Derivation & Independent Recalculation**:
  - In-plane pixel spacing is dx = dy = 0.9765625 mm (for LUNG1-001, 004, 007) or 0.977 mm (for LUNG1-002, 003, 005, 006). Slice thickness dz = 3.0 mm.
  - Slice tumor area: tumor_pixels * (dx * dy).
  - Total 3D Gross Tumor Volume: sum(tumor_voxels) * (dx * dy * dz) in mm^3 and cm^3.
  - All volumes recalculated independently from saved masks match reported values within precision rounding (<0.005 cm^3).
- **Patient-Level Data Leakage Prevention**:
  - Partitioned strictly at the **patient level**:
    - **Train (71.4% of patients, 557 slices)**: LUNG1-001, LUNG1-002, LUNG1-003, LUNG1-004, LUNG1-005
    - **Validation (14.3% of patients, 114 slices)**: LUNG1-006
    - **Test (14.3% of patients, 129 slices)**: LUNG1-007
  - **Zero cross-contamination**: No slices from LUNG1-006 or LUNG1-007 exist in the training partition.

---

### 2.3 Temporal Biomarker Data Cleaning & Horizon Audit

- **Original Candidate Pool**: 106 patients enrolled in the prospective INSPIRE trial (Bratman, Yang et al., Nature Cancer 2020).
- **Filtering & Deduplication**:
  - 12 patients lacked serial follow-up; 94 patients had serial quantified liquid biopsy records.
  - Audited 315 observations: identified and removed 1 exact duplicate record (patient INS-D-001 at week 59.9 / day 419).
  - Final clean count: **315 observations across 94 patients**.
  - Multiple longitudinal timepoints per patient:
    - >= 2 time points: **74 patients**
    - >= 3 time points: **30 patients**
    - >= 4 time points: **21 patients**
    - Maximum time points: **12 visits**
- **Chronological Verification**:
  - Every patient sequence was sorted strictly by days_from_baseline and verified to be monotonically non-decreasing.
  - Zero out-of-order time points.
- **Biomarker Fields**:
  - Standardized quantitative ctDNA concentration: ctdna_value (mean mutant molecules per mL).
  - Standardized variant allele frequency: vaf (mean VAF %).
  - Baseline tissue protein marker: protein_marker (PD-L1 MPS baseline %, available in 312/315 rows).
  - Clinical response: RECIST 1.1 best overall response (CR, PR, SD, PD, NE).
  - Survival outcome: Long-term overall survival vital status (DECEASED, LIVING).
  - Anatomical tumor volume in blood assay: Explicitly recorded as None / empty, as liquid biopsies measure circulating cell-free DNA rather than physical caliper volume.
- **3-Month Horizon Feasibility**:
  - Patients with baseline (-14 to +14 days): 89.
  - Patients with observation at Cycle 3 landmark (~6 weeks / 35-65 days): 71.
  - Patients with observation near 3 months (60-120 days): 29.
  - Recommendation: Cycle 3 is the clinically validated, high-powered landmark; 3-month windows should be modeled as interval targets (60-120 days).

---

## 3. Sample Count & Cleaning Audit Summary Table

| Modality | Original Acquired Count | Excluded Count | Exclusion Reason | Final Clean Count | Integrity Check Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Histopathology Images** | 13,850 candidate patches | 109 edge patches | Non-standard dimensions (50 x <50) | 2,500 curated patches | 100% valid 50x50 RGB PNGs; 0 duplicates |
| **Histopathology Labels** | 13,850 records | 11,350 records | Subsetting to target balance | 2,500 records | 0 missing labels; 100% match to images |
| **Radiology CT Slices** | 800 axial slices (7 pts) | 0 slices | None (complete contiguous scans) | 800 CT slices | 100% valid 512x512 PNGs; 0 corruptions |
| **Radiology GTV Masks** | 800 axial masks (7 pts) | 0 masks | None (complete contiguous masks) | 800 mask slices | 100% paired to images; 0 orphan masks |
| **Radiology Metadata** | 800 records | 0 records | Complete clinical & volume data | 800 records | 0 missing primary keys; patient-level split |
| **Longitudinal Biomarkers**| 316 candidate records before removal of 1 exact duplicate; 315 final records | 1 duplicate row | Duplicate visit (INS-D-001 day 419) | 315 serial records (94 pts)| 100% valid chronological order; 0 duplicate visits |

---

## 4. Known Scientific Limitations

1. **Cross-Modality Patient Linking**:
   - The histopathology images originate from a breast IDC series; radiology CT scans originate from the TCIA NSCLC cohort; and longitudinal biomarkers originate from the prospective INSPIRE trial.
   - **Crucial Rule**: These distinct cohorts are NOT falsely linked to each other at the patient level.
2. **Histopathology Scope**:
   - Breast IDC primary tissue patches; NOT lymph node micro-metastases (CAMELYON16) nor surgical margin-ink margins. Serves as a valid pathology classification proxy.
3. **Radiology Cohort Scale**:
   - 7 patients, 800 slices. Slices represent repeated z-axis observations. Suitable for segmentation/volumetrics; not for population generalization.
4. **Biomarker Forecasting**:
   - 29 patients support a 60-120 day window; 71 patients support a Cycle 3 (35-65 day) window.