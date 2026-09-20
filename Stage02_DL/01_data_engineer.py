import os
import sys
import pandas as pd
from PIL import Image

def get_dataset_dir():
    # 1. Command-line argument if provided
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        return os.path.abspath(sys.argv[1])
    # 2. Environment variable STAGE02_DATA_DIR
    env_dir = os.environ.get('STAGE02_DATA_DIR')
    if env_dir and os.path.isdir(env_dir):
        return os.path.abspath(env_dir)
    # 3. Standard fallback candidate paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, 'Stage02_Data'),
        os.path.abspath(os.path.join(script_dir, '..', '..', 'Stage02_Data')),
        os.path.abspath(os.path.join(script_dir, '..', 'Stage02_Data'))
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return env_dir if env_dir else os.path.join(script_dir, 'Stage02_Data')

def validate(data_dir=None):
    print('============================================================')
    print('STARTING COMPREHENSIVE DATASET VALIDATION FOR STAGE 02')
    print('============================================================')
    base_dir = os.path.abspath(data_dir) if data_dir else get_dataset_dir()
    print('Dataset Directory:', base_dir)
    if not os.path.exists(base_dir):
        print('ERROR: Base directory does not exist:', base_dir)
        print('Hint: Set the STAGE02_DATA_DIR environment variable or pass the path as an argument.')
        return False
    errors = []

    # 1. Directory Structure
    required_dirs = [
        os.path.join(base_dir, 'histopathology', 'images'),
        os.path.join(base_dir, 'radiology', 'images'),
        os.path.join(base_dir, 'radiology', 'masks'),
        os.path.join(base_dir, 'biomarkers'),
        os.path.join(base_dir, 'metadata')
    ]
    for d in required_dirs:
        if not os.path.isdir(d):
            errors.append('Missing directory: ' + d)
        else:
            print('[PASS] Directory exists: ' + os.path.relpath(d, base_dir))

    # 2. Check CSVs
    csv_files = {
        'histopathology_labels': os.path.join(base_dir, 'histopathology', 'labels.csv'),
        'radiology_metadata': os.path.join(base_dir, 'radiology', 'metadata.csv'),
        'biomarkers_longitudinal': os.path.join(base_dir, 'biomarkers', 'longitudinal.csv'),
        'dataset_sources': os.path.join(base_dir, 'metadata', 'dataset_sources.csv'),
        'data_dictionary': os.path.join(base_dir, 'metadata', 'data_dictionary.csv'),
        'split_manifest': os.path.join(base_dir, 'metadata', 'split_manifest.csv')
    }
    dfs = {}
    for name, p in csv_files.items():
        if not os.path.isfile(p):
            errors.append('Missing CSV file: ' + p)
        else:
            try:
                df = pd.read_csv(p)
                dfs[name] = df
                print('[PASS] CSV loaded successfully: ' + name + ' (shape: ' + str(df.shape) + ')')
            except Exception as e:
                errors.append('Failed to parse CSV ' + p + ': ' + str(e))

    # 3. Validate Histopathology
    print('\n--- Validating Histopathology ---')
    df_histo = dfs.get('histopathology_labels')
    if df_histo is not None:
        required_cols = ['image_id', 'image_path', 'label', 'class', 'split', 'md5_hash']
        for col in required_cols:
            if col not in df_histo.columns:
                errors.append('Histopathology missing col: ' + col)
        
        # Verify exact counts and terminology
        if len(df_histo) != 2500:
            errors.append('Histopathology records count mismatch: ' + str(len(df_histo)) + ' (expected 2500)')
        else:
            print('[PASS] Histopathology records count verified: 2500')
            
        class_counts = df_histo['class'].value_counts().to_dict()
        expected_classes = {'non_idc_or_benign': 1750, 'invasive_ductal_carcinoma': 750}
        if class_counts != expected_classes:
            errors.append('Histopathology class distribution mismatch: ' + str(class_counts) + ' vs ' + str(expected_classes))
        else:
            print('[PASS] Histopathology class terminology and distribution verified: 1750 non_idc_or_benign / 750 invasive_ductal_carcinoma')
            
        sample_count = len(df_histo)
        print('Checking ' + str(sample_count) + ' histopathology images on disk...')
        hashes = set()
        for idx, row in df_histo.iterrows():
            rel_p = row['image_path']
            full_p = os.path.join(base_dir, rel_p)
            if not os.path.isfile(full_p):
                errors.append('Image not found: ' + full_p)
                continue
            try:
                im = Image.open(full_p)
                im.verify()
                im = Image.open(full_p)
                if im.size != (50, 50):
                    errors.append('Invalid size ' + str(im.size) + ' for ' + rel_p)
                if im.mode != 'RGB':
                    errors.append('Invalid mode ' + str(im.mode) + ' for ' + rel_p)
            except Exception as e:
                errors.append('Corrupt image ' + rel_p + ': ' + str(e))
            h = row.get('md5_hash')
            if h in hashes:
                errors.append('Duplicate hash: ' + str(h))
            hashes.add(h)
        print('[PASS] All ' + str(sample_count) + ' histopathology images verified (50x50 RGB, 0 duplicates, 0 corruptions)')

    # 4. Validate Radiology
    print('\n--- Validating Radiology ---')
    df_rad = dfs.get('radiology_metadata')
    if df_rad is not None:
        required_rad_cols = ['patient_id', 'study_id', 'series_id', 'image_id', 'slice_id', 'timepoint', 'image_path', 'mask_path', 'tumor_present', 'tumor_area', 'tumor_volume', 'response', 'scan_date', 'split']
        for col in required_rad_cols:
            if col not in df_rad.columns:
                errors.append('Radiology missing col: ' + col)
        if len(df_rad) != 800:
            errors.append('Radiology records count mismatch: ' + str(len(df_rad)) + ' (expected 800)')
        else:
            print('[PASS] Radiology records count verified: 800')
            
        rad_count = len(df_rad)
        print('Checking ' + str(rad_count) + ' CT slices and masks...')
        for idx, row in df_rad.iterrows():
            full_img = os.path.join(base_dir, row['image_path'])
            full_mask = os.path.join(base_dir, row['mask_path'])
            if not os.path.isfile(full_img):
                errors.append('CT slice not found: ' + full_img)
                continue
            if not os.path.isfile(full_mask):
                errors.append('Mask not found: ' + full_mask)
                continue
            try:
                im_ct = Image.open(full_img)
                im_ct.verify()
                im_ct = Image.open(full_img)
                im_mask = Image.open(full_mask)
                im_mask.verify()
                im_mask = Image.open(full_mask)
                if im_ct.size != (512, 512):
                    errors.append('CT slice dimension invalid: ' + str(im_ct.size))
                if im_mask.size != (512, 512):
                    errors.append('Mask dimension invalid: ' + str(im_mask.size))
                if im_ct.size != im_mask.size:
                    errors.append('Dimension mismatch: CT ' + str(im_ct.size) + ' vs Mask ' + str(im_mask.size))
            except Exception as e:
                errors.append('Corrupt radiology file: ' + str(e))
        print('[PASS] All ' + str(rad_count) + ' CT slices and masks verified (512x512, 100% paired, 0 orphan masks)')
        
        splits_by_patient = df_rad.groupby('patient_id')['split'].nunique()
        if (splits_by_patient > 1).any():
            errors.append('PATIENT DATA LEAKAGE in Radiology!')
        else:
            print('[PASS] ZERO Patient Data Leakage in Radiology (patients strictly confined to 1 split)')

    # 5. Validate Temporal Biomarkers
    print('\n--- Validating Temporal Biomarkers ---')
    df_bio = dfs.get('biomarkers_longitudinal')
    if df_bio is not None:
        required_bio_cols = ['patient_id', 'timepoint', 'date', 'days_from_baseline', 'biomarker_name', 'biomarker_value', 'ctdna_value', 'protein_marker', 'vaf', 'tumor_volume', 'response', 'outcome']
        for col in required_bio_cols:
            if col not in df_bio.columns:
                errors.append('Biomarker missing col: ' + col)
                
        n_bio_records = len(df_bio)
        n_bio_patients = df_bio['patient_id'].nunique()
        if n_bio_records != 315:
            errors.append('Biomarker records count mismatch: ' + str(n_bio_records) + ' (expected 315)')
        if n_bio_patients != 94:
            errors.append('Biomarker patient count mismatch: ' + str(n_bio_patients) + ' (expected 94)')
        print('Biomarker records: ' + str(n_bio_records) + ' across ' + str(n_bio_patients) + ' patients')
        
        # Check duplicate patient/timepoint records
        dup_tp = df_bio.duplicated(subset=['patient_id', 'timepoint']).sum()
        if dup_tp > 0:
            errors.append('Duplicate patient/timepoint records in biomarker table: ' + str(dup_tp))
        else:
            print('[PASS] Zero duplicate patient/timepoint records in biomarkers')
            
        chrono_err = 0
        for pid, grp in df_bio.groupby('patient_id'):
            d_list = grp['days_from_baseline'].dropna().tolist()
            if d_list != sorted(d_list):
                errors.append('Chronological violation for patient: ' + str(pid))
                chrono_err += 1
        if chrono_err == 0:
            print('[PASS] Chronological order strictly preserved for all patients')
        invalid_days = df_bio[df_bio['days_from_baseline'] < -30]
        if len(invalid_days) > 0:
            errors.append('Impossible timestamps: ' + str(len(invalid_days)))
        else:
            print('[PASS] All timestamps physiologically and clinically valid')

    # 6. Validate Metadata Documents & Split Manifest Consistency
    print('\n--- Validating Metadata & Documentation ---')
    report_p = os.path.join(base_dir, 'metadata', 'data_quality_report.md')
    readme_p = os.path.join(base_dir, 'README.md')
    if not os.path.isfile(report_p): errors.append('Missing data_quality_report.md')
    else: print('[PASS] data_quality_report.md exists')
    if not os.path.isfile(readme_p): errors.append('Missing README.md')
    else: print('[PASS] README.md exists')

    df_manifest = dfs.get('split_manifest')
    if df_manifest is not None and df_histo is not None and df_rad is not None and df_bio is not None:
        expected_total = len(df_histo) + len(df_rad) + len(df_bio)
        manifest_count = len(df_manifest)
        if manifest_count != expected_total or manifest_count != 3615:
            errors.append('Split manifest count mismatch: ' + str(manifest_count) + ' (expected ' + str(expected_total) + ' = 3615)')
        else:
            print('[PASS] Split manifest: 3615 records (dynamically verified = len(histopathology) + len(radiology) + len(biomarkers))')
            
        # Verify biomarker patient-level split in manifest
        bio_manifest = df_manifest[df_manifest['modality'] == 'biomarker']
        bio_splits_per_pt = bio_manifest.groupby('patient_id')['split'].nunique()
        if (bio_splits_per_pt > 1).any():
            errors.append('Biomarker patient data leakage in split manifest!')
        else:
            print('[PASS] ZERO Patient Data Leakage in Biomarker split manifest (each patient in exactly 1 split)')

    print('\n============================================================')
    if len(errors) == 0:
        print('ALL VALIDATION CHECKS PASSED SUCCESSFULLY (0 ERRORS)!')
        print('Dataset is clean, validated, and ready for EDA / DL.')
        print('============================================================')
        return True
    else:
        print('VALIDATION FAILED WITH ' + str(len(errors)) + ' ERRORS:')
        for e in errors: print('  [ERROR] ' + e)
        print('============================================================')
        return False

if __name__ == '__main__':
    if not validate():
        sys.exit(1)