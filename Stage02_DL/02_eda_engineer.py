import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from PIL import Image

# Set plotting aesthetics
sns.set_theme(style='whitegrid')
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.titlesize': 14,
    'figure.dpi': 150
})

def get_dataset_dir(cli_arg=None):
    """
    Resolve dataset directory using prioritized discovery:
    1. CLI argument if passed
    2. STAGE02_DATA_DIR environment variable
    3. Standard relative candidates
    """
    if cli_arg and os.path.isdir(cli_arg):
        return os.path.abspath(cli_arg)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and os.path.isdir(sys.argv[1]):
        return os.path.abspath(sys.argv[1])
    env_dir = os.environ.get('STAGE02_DATA_DIR')
    if env_dir and os.path.isdir(env_dir):
        return os.path.abspath(env_dir)
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

def setup_output_dirs(base_output_dir):
    histo_out = os.path.join(base_output_dir, 'histopathology')
    rad_out = os.path.join(base_output_dir, 'radiology')
    bio_out = os.path.join(base_output_dir, 'biomarkers')
    for d in [histo_out, rad_out, bio_out]:
        os.makedirs(d, exist_ok=True)
    return histo_out, rad_out, bio_out

# ==============================================================================
# 1. HISTOPATHOLOGY EDA
# ==============================================================================
def run_histopathology_eda(dataset_dir, output_dir):
    print("\n" + "="*70)
    print("RUNNING HISTOPATHOLOGY EDA (2,500 Curated Patches)")
    print("="*70)
    
    labels_csv = os.path.join(dataset_dir, 'histopathology', 'labels.csv')
    if not os.path.isfile(labels_csv):
        raise FileNotFoundError(f"Histopathology labels not found at: {labels_csv}")
    
    df_h = pd.read_csv(labels_csv)
    total_samples = len(df_h)
    print(f"Total Histopathology samples: {total_samples}")
    
    # Class distribution
    class_counts = df_h['class'].value_counts()
    class_props = df_h['class'].value_counts(normalize=True) * 100.0
    print("\nClass distribution:")
    for cls_name, cnt in class_counts.items():
        print(f" - {cls_name}: {cnt} ({class_props[cls_name]:.2f}%)")
        
    # Split distribution & class distribution per split
    split_dist = pd.crosstab(df_h['split'], df_h['class'], margins=True)
    split_props = pd.crosstab(df_h['split'], df_h['class'], normalize='index') * 100.0
    print("\nSplit Cross-tabulation:\n", split_dist)
    
    # Image integrity, dimensions, channels, hash check
    print("\nAuditing image integrity, dimensions, and pixel statistics...")
    dims_set = set()
    channels_set = set()
    hashes_list = []
    
    r_means, g_means, b_means = [], [], []
    r_stds, g_stds, b_stds = [], [], []
    brightness_list, contrast_list = [], []
    
    for idx, row in df_h.iterrows():
        img_p = os.path.join(dataset_dir, row['image_path'])
        with Image.open(img_p) as im:
            dims_set.add(im.size)
            channels_set.add(im.mode)
            arr = np.array(im, dtype=np.float32)
            
        r_c = arr[:, :, 0]
        g_c = arr[:, :, 1]
        b_c = arr[:, :, 2]
        
        r_means.append(float(r_c.mean()))
        g_means.append(float(g_c.mean()))
        b_means.append(float(b_c.mean()))
        
        r_stds.append(float(r_c.std()))
        g_stds.append(float(g_c.std()))
        b_stds.append(float(b_c.std()))
        
        # Standard relative luminance / perceptual brightness: 0.2989*R + 0.5870*G + 0.1140*B
        gray = 0.2989 * r_c + 0.5870 * g_c + 0.1140 * b_c
        brightness_list.append(float(gray.mean()))
        contrast_list.append(float(gray.std()))
        
        hashes_list.append(row.get('md5_hash', ''))

    df_h['r_mean'] = r_means
    df_h['g_mean'] = g_means
    df_h['b_mean'] = b_means
    df_h['r_std'] = r_stds
    df_h['g_std'] = g_stds
    df_h['b_std'] = b_stds
    df_h['brightness'] = brightness_list
    df_h['contrast'] = contrast_list
    
    unique_hashes = len(set(hashes_list))
    duplicate_count = total_samples - unique_hashes
    print(f"Dimensions found across dataset: {dims_set}")
    print(f"Color modes found: {channels_set}")
    print(f"Unique MD5 image hashes: {unique_hashes} (Duplicates: {duplicate_count})")
    
    # Save descriptive pixel stats table
    stats_by_class = df_h.groupby('class')[['r_mean', 'g_mean', 'b_mean', 'brightness', 'contrast']].agg(['mean', 'std', 'min', 'max'])
    stats_by_class_path = os.path.join(output_dir, 'histopathology_pixel_stats.csv')
    stats_by_class.to_csv(stats_by_class_path)
    
    summary_split_path = os.path.join(output_dir, 'histopathology_split_summary.csv')
    split_dist.to_csv(summary_split_path)
    print(f"Saved Histopathology summary tables to {output_dir}")

    # --- PLOT 1: Class Distribution & Proportions ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    palette = {'non_idc_or_benign': '#4A90E2', 'invasive_ductal_carcinoma': '#E74C3C'}
    
    sns.countplot(data=df_h, x='class', hue='class', palette=palette, ax=axes[0], legend=False)
    axes[0].set_title('Histopathology Class Counts (Total = 2,500)')
    axes[0].set_xlabel('Pathology Classification')
    axes[0].set_ylabel('Sample Count')
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}\n({p.get_height()/total_samples*100:.1f}%)",
                         (p.get_x() + p.get_width() / 2., p.get_height() / 2.),
                         ha='center', va='center', color='white', fontweight='bold')
                         
    # Split class distribution
    df_split_plot = df_h.groupby(['split', 'class']).size().reset_index(name='count')
    sns.barplot(data=df_split_plot, x='split', y='count', hue='class', palette=palette, ax=axes[1])
    axes[1].set_title('Class Distribution Across Train / Val / Test Splits')
    axes[1].set_xlabel('Split Partition')
    axes[1].set_ylabel('Sample Count')
    for p in axes[1].patches:
        h = p.get_height()
        if h > 0:
            axes[1].annotate(f"{int(h)}",
                             (p.get_x() + p.get_width() / 2., h + 15),
                             ha='center', va='bottom', fontsize=8)
    axes[1].legend(title='Class')
    plt.tight_layout()
    plot1_p = os.path.join(output_dir, '01_histopathology_class_and_split_dist.png')
    plt.savefig(plot1_p)
    plt.close()
    
    # --- PLOT 2: Pixel Intensity & Channel Distribution ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    channels = [('r_mean', 'Red Channel Mean', '#C0392B'),
                ('g_mean', 'Green Channel Mean', '#27AE60'),
                ('b_mean', 'Blue Channel Mean', '#2980B9')]
    for ax, (col, title, color) in zip(axes, channels):
        for cls_name, cls_color in palette.items():
            subset = df_h[df_h['class'] == cls_name][col]
            sns.kdeplot(subset, ax=ax, label=cls_name, color=cls_color, fill=True, alpha=0.3, linewidth=2)
        ax.set_title(f'{title} by Class')
        ax.set_xlabel('Mean Intensity (0-255)')
        ax.set_ylabel('Density')
        ax.legend(title='Class')
    plt.tight_layout()
    plot2_p = os.path.join(output_dir, '02_histopathology_rgb_channel_distributions.png')
    plt.savefig(plot2_p)
    plt.close()

    # --- PLOT 3: Brightness vs Contrast Analysis ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=df_h, x='brightness', y='contrast', hue='class', palette=palette, alpha=0.5, s=25, ax=axes[0])
    axes[0].set_title('Patch Luminance (Brightness) vs Texture Contrast')
    axes[0].set_xlabel('Mean Luminance (Gray Level)')
    axes[0].set_ylabel('Contrast (Std Dev of Gray Level)')
    
    sns.violinplot(data=df_h, x='class', y='brightness', hue='class', palette=palette, ax=axes[1], inner='quartile', legend=False)
    axes[1].set_title('Brightness Distribution by Class')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Luminance Value')
    plt.tight_layout()
    plot3_p = os.path.join(output_dir, '03_histopathology_brightness_contrast.png')
    plt.savefig(plot3_p)
    plt.close()

    # --- PLOT 4: Representative Montage Grid (by Class and Split) ---
    np.random.seed(42)
    fig, axes = plt.subplots(4, 6, figsize=(14, 10))
    fig.suptitle('Representative Histopathology Patches (50x50 RGB) Across Splits & Classes', fontsize=14, y=0.98)
    
    # Rows 0-1: Class 0 (non_idc_or_benign) across Train, Val, Test
    # Rows 2-3: Class 1 (invasive_ductal_carcinoma) across Train, Val, Test
    categories = [
        ('non_idc_or_benign', 'train'),
        ('non_idc_or_benign', 'val'),
        ('non_idc_or_benign', 'test'),
        ('invasive_ductal_carcinoma', 'train'),
        ('invasive_ductal_carcinoma', 'val'),
        ('invasive_ductal_carcinoma', 'test')
    ]
    
    row_idx = 0
    col_idx = 0
    for cls_name in ['non_idc_or_benign', 'invasive_ductal_carcinoma']:
        for split_name in ['train', 'val', 'test']:
            sample_rows = df_h[(df_h['class'] == cls_name) & (df_h['split'] == split_name)].sample(4, random_state=42)
            for _, s_row in sample_rows.iterrows():
                im_p = os.path.join(dataset_dir, s_row['image_path'])
                im = Image.open(im_p)
                ax = axes[row_idx, col_idx]
                ax.imshow(im)
                ax.axis('off')
                short_cls = "Non-IDC" if cls_name == 'non_idc_or_benign' else "IDC (Pos)"
                ax.set_title(f"{short_cls}\n[{split_name.upper()}]", fontsize=8)
                col_idx += 1
                if col_idx >= 6:
                    col_idx = 0
                    row_idx += 1
                    if row_idx >= 4:
                        break
            if row_idx >= 4:
                break
    plt.tight_layout()
    plot4_p = os.path.join(output_dir, '04_histopathology_representative_patches.png')
    plt.savefig(plot4_p)
    plt.close()
    
    print("Histopathology EDA completed successfully.")
    return {
        'total_samples': total_samples,
        'class_distribution': class_counts.to_dict(),
        'class_proportions': class_props.to_dict(),
        'unique_hashes': unique_hashes,
        'dimensions': list(dims_set),
        'channels': list(channels_set),
        'mean_rgb_idc': [float(stats_by_class.loc['invasive_ductal_carcinoma', ('r_mean', 'mean')]),
                         float(stats_by_class.loc['invasive_ductal_carcinoma', ('g_mean', 'mean')]),
                         float(stats_by_class.loc['invasive_ductal_carcinoma', ('b_mean', 'mean')])],
        'mean_rgb_non_idc': [float(stats_by_class.loc['non_idc_or_benign', ('r_mean', 'mean')]),
                             float(stats_by_class.loc['non_idc_or_benign', ('g_mean', 'mean')]),
                             float(stats_by_class.loc['non_idc_or_benign', ('b_mean', 'mean')])]
    }

# ==============================================================================
# 2. RADIOLOGY EDA
# ==============================================================================
def run_radiology_eda(dataset_dir, output_dir):
    print("\n" + "="*70)
    print("RUNNING RADIOLOGY EDA (800 Axial CT Slices across 7 Patients)")
    print("="*70)
    
    rad_csv = os.path.join(dataset_dir, 'radiology', 'metadata.csv')
    if not os.path.isfile(rad_csv):
        raise FileNotFoundError(f"Radiology metadata not found at: {rad_csv}")
        
    df_rad = pd.read_csv(rad_csv)
    total_slices = len(df_rad)
    unique_patients = df_rad['patient_id'].nunique()
    print(f"CRITICAL COHORT STRUCTURE: {total_slices} CT slices across strictly {unique_patients} patients.")
    print("WARNING: Slices represent repeated axial z-axis scans of 7 patients, NOT 800 independent patients.")
    
    # Slices per patient
    patient_summary = df_rad.groupby('patient_id').agg(
        total_slices=('slice_id', 'count'),
        tumor_slices=('tumor_present', 'sum'),
        tumor_volume_cm3=('tumor_volume', 'first'),
        overall_stage=('overall_stage', 'first'),
        histology=('histology', 'first'),
        survival_time_days=('survival_time_days', 'first'),
        outcome_2yr_mortality=('outcome_2yr_mortality', 'first'),
        split=('split', 'first')
    ).reset_index()
    patient_summary['tumor_slice_ratio_pct'] = (patient_summary['tumor_slices'] / patient_summary['total_slices']) * 100.0
    print("\nPatient-Level Summary Table:\n", patient_summary[['patient_id', 'split', 'total_slices', 'tumor_slices', 'tumor_volume_cm3', 'overall_stage']])

    # Split distribution (patients and slices)
    split_pt_counts = patient_summary.groupby('split')['patient_id'].count()
    split_slice_counts = df_rad.groupby('split')['slice_id'].count()
    print(f"\nSplit Distribution: Patients = {split_pt_counts.to_dict()} | Slices = {split_slice_counts.to_dict()}")

    # Image and mask integrity, dimensions, mask coverage
    print("\nAuditing CT slices, masks, and calculating area/coverage metrics...")
    ct_dims = set()
    mask_dims = set()
    mask_coverages = []
    
    for idx, row in df_rad.iterrows():
        ct_p = os.path.join(dataset_dir, row['image_path'])
        mask_p = os.path.join(dataset_dir, row['mask_path'])
        
        with Image.open(ct_p) as im_ct:
            ct_dims.add(im_ct.size)
        with Image.open(mask_p) as im_m:
            mask_dims.add(im_m.size)
            if row['tumor_present'] == 1:
                arr_m = np.array(im_m)
                pos_pixels = np.count_nonzero(arr_m > 0)
                coverage_pct = (pos_pixels / (im_m.size[0] * im_m.size[1])) * 100.0
                mask_coverages.append(coverage_pct)
            else:
                mask_coverages.append(0.0)
                
    df_rad['mask_coverage_pct'] = mask_coverages
    
    # Tumor-positive vs negative slices
    tumor_slice_counts = df_rad['tumor_present'].value_counts()
    print(f"Tumor-positive slices: {tumor_slice_counts.get(1, 0)} ({tumor_slice_counts.get(1, 0)/total_slices*100:.1f}%)")
    print(f"Tumor-negative slices: {tumor_slice_counts.get(0, 0)} ({tumor_slice_counts.get(0, 0)/total_slices*100:.1f}%)")

    # Slice area statistics on positive slices
    pos_slices = df_rad[df_rad['tumor_present'] == 1]
    area_stats = pos_slices['tumor_area'].describe()
    cov_stats = pos_slices['mask_coverage_pct'].describe()
    print("\nTumor cross-sectional area (mm^2) on tumor-positive slices:\n", area_stats)
    print("\nMask coverage (%) on tumor-positive slices:\n", cov_stats)

    # Save summary tables
    patient_summary_path = os.path.join(output_dir, 'radiology_patient_summary.csv')
    patient_summary.to_csv(patient_summary_path, index=False)
    
    slice_stats_path = os.path.join(output_dir, 'radiology_slice_metrics.csv')
    df_rad[['patient_id', 'slice_id', 'split', 'tumor_present', 'tumor_area', 'mask_coverage_pct']].to_csv(slice_stats_path, index=False)
    print(f"Saved Radiology summary tables to {output_dir}")

    # --- PLOT 1: Patient Slice Breakdown and Tumor Burden ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_pts = patient_summary['patient_id']
    axes[0].bar(x_pts, patient_summary['total_slices'], label='Tumor-Negative Slices', color='#BDC3C7')
    axes[0].bar(x_pts, patient_summary['tumor_slices'], label='Tumor-Positive Slices (GTV-1)', color='#E74C3C')
    axes[0].set_title('CT Slices per Patient (Total = 800 across 7 Patients)')
    axes[0].set_xlabel('Patient Identifier')
    axes[0].set_ylabel('Number of Axial Slices')
    axes[0].legend()
    for idx, row in patient_summary.iterrows():
        axes[0].text(idx, row['total_slices'] + 2, f"Total:{row['total_slices']}\nGTV:{row['tumor_slices']}",
                     ha='center', va='bottom', fontsize=8)
                     
    # Slices and Patient Split Distribution
    split_summary = df_rad.groupby(['split']).agg(
        slices=('slice_id', 'count'),
        patients=('patient_id', 'nunique'),
        tumor_slices=('tumor_present', 'sum')
    ).reset_index()
    
    axes[1].bar(split_summary['split'], split_summary['slices'], color=['#3498DB', '#F39C12', '#2ECC71'])
    axes[1].set_title('Radiology Partitioning (Patient-Level Split)')
    axes[1].set_xlabel('Dataset Split')
    axes[1].set_ylabel('Slice Count')
    for idx, row in split_summary.iterrows():
        axes[1].text(idx, row['slices']/2, f"{row['slices']} Slices\n({row['patients']} Patients)\nGTV: {row['tumor_slices']} slices",
                     ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    plt.tight_layout()
    plot1_p = os.path.join(output_dir, '01_radiology_patient_and_split_breakdown.png')
    plt.savefig(plot1_p)
    plt.close()

    # --- PLOT 2: Tumor Cross-Sectional Area and Coverage Distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(pos_slices['tumor_area'], kde=True, color='#E74C3C', bins=20, ax=axes[0])
    axes[0].axvline(pos_slices['tumor_area'].median(), color='black', linestyle='--', label=f"Median: {pos_slices['tumor_area'].median():.1f} mm²")
    axes[0].set_title('Tumor Cross-Sectional Area Distribution (157 Positive Slices)')
    axes[0].set_xlabel('Tumor Area (mm²)')
    axes[0].set_ylabel('Slice Count')
    axes[0].legend()
    
    sns.boxplot(data=pos_slices, x='patient_id', y='tumor_area', palette='Set2', ax=axes[1])
    axes[1].set_title('Tumor Area Variance by Patient')
    axes[1].set_xlabel('Patient ID')
    axes[1].set_ylabel('Tumor Area (mm²)')
    plt.tight_layout()
    plot2_p = os.path.join(output_dir, '02_radiology_tumor_area_distributions.png')
    plt.savefig(plot2_p)
    plt.close()

    # --- PLOT 3: 3D Gross Tumor Volume vs Clinical Stage & Outcomes ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=patient_summary, x='patient_id', y='tumor_volume_cm3', hue='overall_stage', dodge=False, palette='Dark2', ax=axes[0])
    axes[0].set_title('Patient 3D Gross Tumor Volume (GTV-1 cm³) by Clinical Stage')
    axes[0].set_xlabel('Patient ID')
    axes[0].set_ylabel('Gross Tumor Volume (cm³)')
    axes[0].legend(title='Stage')
    for idx, row in patient_summary.iterrows():
        axes[0].text(idx, row['tumor_volume_cm3'] + 5, f"{row['tumor_volume_cm3']:.1f} cm³",
                     ha='center', va='bottom', fontsize=9)

    # Mask coverage % on positive slices
    sns.histplot(pos_slices['mask_coverage_pct'], kde=True, color='#8E44AD', bins=20, ax=axes[1])
    axes[1].set_title('Mask Coverage % on 512x512 Slice Canvas (Sparse Signal)')
    axes[1].set_xlabel('Tumor Area Coverage (% of 512x512 pixels)')
    axes[1].set_ylabel('Positive Slice Count')
    axes[1].axvline(pos_slices['mask_coverage_pct'].mean(), color='black', linestyle=':', label=f"Mean: {pos_slices['mask_coverage_pct'].mean():.2f}%")
    axes[1].legend()
    plt.tight_layout()
    plot3_p = os.path.join(output_dir, '03_radiology_gtv_volumes_and_coverage.png')
    plt.savefig(plot3_p)
    plt.close()

    # --- PLOT 4: Representative CT Slices and Paired Mask Overlay ---
    # Pick 3 representative patients with substantial tumors: e.g. LUNG1-001, LUNG1-002, LUNG1-004
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('Representative Axial CT Slices (512x512) and GTV-1 Delineations', fontsize=14)
    rep_patients = ['LUNG1-001', 'LUNG1-002', 'LUNG1-004']
    
    for i, pid in enumerate(rep_patients):
        # Find slice with maximum tumor area for this patient
        pt_pos = df_rad[(df_rad['patient_id'] == pid) & (df_rad['tumor_present'] == 1)]
        max_slice_row = pt_pos.sort_values('tumor_area', ascending=False).iloc[0]
        
        ct_p = os.path.join(dataset_dir, max_slice_row['image_path'])
        mask_p = os.path.join(dataset_dir, max_slice_row['mask_path'])
        
        ct_im = Image.open(ct_p).convert('L')
        mask_im = Image.open(mask_p).convert('L')
        
        ct_arr = np.array(ct_im)
        mask_arr = np.array(mask_im)
        
        # Row 0: Pure CT
        axes[0, i].imshow(ct_arr, cmap='gray')
        axes[0, i].set_title(f"{pid} (Slice {max_slice_row['slice_id']})\nStage: {max_slice_row['overall_stage']}")
        axes[0, i].axis('off')
        
        # Row 1: CT with Red Mask Overlay
        axes[1, i].imshow(ct_arr, cmap='gray')
        masked_overlay = np.ma.masked_where(mask_arr == 0, mask_arr)
        axes[1, i].imshow(masked_overlay, cmap='autumn', alpha=0.55)
        axes[1, i].set_title(f"GTV-1 Overlay (Area: {max_slice_row['tumor_area']:.1f} mm²)\nSplit: {max_slice_row['split'].upper()}")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plot4_p = os.path.join(output_dir, '04_radiology_representative_ct_and_masks.png')
    plt.savefig(plot4_p)
    plt.close()

    print("Radiology EDA completed successfully.")
    return {
        'total_slices': total_slices,
        'unique_patients': unique_patients,
        'patient_list': patient_summary['patient_id'].tolist(),
        'slices_per_patient': dict(zip(patient_summary['patient_id'], patient_summary['total_slices'])),
        'tumor_positive_slices': int(tumor_slice_counts.get(1, 0)),
        'tumor_negative_slices': int(tumor_slice_counts.get(0, 0)),
        'dimensions': list(ct_dims),
        'mean_tumor_area_mm2': float(area_stats['mean']),
        'median_tumor_area_mm2': float(area_stats['50%']),
        'mean_mask_coverage_pct': float(cov_stats['mean']),
        'patient_volumes_cm3': dict(zip(patient_summary['patient_id'], patient_summary['tumor_volume_cm3']))
    }

# ==============================================================================
# 3. BIOMARKER EDA
# ==============================================================================
def run_biomarker_eda(dataset_dir, output_dir):
    print("\n" + "="*70)
    print("RUNNING BIOMARKER EDA (315 Longitudinal Records across 94 Patients)")
    print("="*70)
    
    bio_csv = os.path.join(dataset_dir, 'biomarkers', 'longitudinal.csv')
    if not os.path.isfile(bio_csv):
        raise FileNotFoundError(f"Biomarker table not found at: {bio_csv}")
        
    df_b = pd.read_csv(bio_csv)
    manifest_csv = os.path.join(dataset_dir, 'metadata', 'split_manifest.csv')
    df_manifest = pd.read_csv(manifest_csv) if os.path.isfile(manifest_csv) else None
    
    # Merge split if not in table
    if 'split' not in df_b.columns and df_manifest is not None:
        bio_splits = df_manifest[df_manifest['modality'] == 'biomarker'].drop_duplicates(subset=['patient_id'])[['patient_id', 'split']]
        df_b = df_b.merge(bio_splits, on='patient_id', how='left')
    
    total_records = len(df_b)
    unique_patients = df_b['patient_id'].nunique()
    print(f"Total serial biomarker observations: {total_records} across {unique_patients} patients.")
    
    # Records per patient
    rec_per_pt = df_b.groupby('patient_id').size()
    print(f"Visits per patient: Min={rec_per_pt.min()}, Median={rec_per_pt.median()}, Mean={rec_per_pt.mean():.2f}, Max={rec_per_pt.max()}")
    
    # Chronological ordering check
    chrono_violations = 0
    for pid, grp in df_b.groupby('patient_id'):
        days = grp['days_from_baseline'].dropna().tolist()
        if days != sorted(days):
            chrono_violations += 1
    print(f"Chronological ordering check: {chrono_violations} violations found.")
    
    # Duplicate visits check
    dup_visits = df_b.duplicated(subset=['patient_id', 'timepoint']).sum()
    print(f"Duplicate patient-timepoint visits: {dup_visits}")
    
    # Missing values analysis
    missing_summary = df_b.isnull().sum()
    print("\nMissing Values per Column:\n", missing_summary[missing_summary > 0])

    # Biomarker distributions
    ctdna_stats = df_b['ctdna_value'].describe()
    vaf_stats = df_b['vaf'].describe()
    pdl1_stats = df_b['protein_marker'].describe()
    print("\nctDNA (molecules/mL) descriptive statistics:\n", ctdna_stats)
    print("\nVAF (fraction) descriptive statistics:\n", vaf_stats)
    print("\nPD-L1 MPS Baseline (%) descriptive statistics:\n", pdl1_stats)
    
    # Baseline vs later visits
    baseline_records = df_b[df_b['days_from_baseline'] <= 0]
    post_baseline_records = df_b[df_b['days_from_baseline'] > 0]
    print(f"Baseline measurements: {len(baseline_records)} records ({baseline_records['patient_id'].nunique()} patients)")
    print(f"Post-baseline longitudinal measurements: {len(post_baseline_records)} records")
    
    # Response and Outcome counts
    response_counts = df_b['response'].value_counts()
    outcome_counts = df_b['outcome'].value_counts()
    print(f"\nRECIST 1.1 Response Distribution:\n{response_counts}")
    print(f"\nOverall Survival Outcome Distribution:\n{outcome_counts}")

    # Save summary tables
    pt_summary = df_b.groupby('patient_id').agg(
        record_count=('timepoint', 'count'),
        baseline_days=('days_from_baseline', 'min'),
        max_days=('days_from_baseline', 'max'),
        baseline_ctdna=('ctdna_value', 'first'),
        baseline_vaf=('vaf', 'first'),
        pdl1_mps=('protein_marker', 'first'),
        response=('response', 'first'),
        outcome=('outcome', 'first'),
        split=('split', 'first') if 'split' in df_b.columns else ('timepoint', 'first')
    ).reset_index()
    pt_summary.to_csv(os.path.join(output_dir, 'biomarkers_patient_summary.csv'), index=False)
    
    marker_dist_summary = pd.DataFrame({
        'ctdna_molecules_per_mL': ctdna_stats,
        'vaf_fraction': vaf_stats,
        'pdl1_mps_percent': pdl1_stats
    })
    marker_dist_summary.to_csv(os.path.join(output_dir, 'biomarkers_distribution_stats.csv'))
    print(f"Saved Biomarker summary tables to {output_dir}")

    # --- PLOT 1: Patient Longitudinal Visit Timelines (Swimmer Plot) ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Subplot A: Histogram of visits per patient
    sns.histplot(rec_per_pt, bins=range(1, rec_per_pt.max() + 2), discrete=True, color='#2980B9', ax=axes[0])
    axes[0].set_title(f'Distribution of Serial Observations per Patient (N = {unique_patients})')
    axes[0].set_xlabel('Number of Serial Longitudinal Visits')
    axes[0].set_ylabel('Number of Patients')
    axes[0].axvline(rec_per_pt.median(), color='#E74C3C', linestyle='--', label=f"Median Visits: {rec_per_pt.median():.0f}")
    axes[0].legend()
    
    # Subplot B: Swimmer plot for top 25 patients with most visits
    top_patients = rec_per_pt.sort_values(ascending=False).head(25).index.tolist()
    df_swimmer = df_b[df_b['patient_id'].isin(top_patients)].copy()
    
    y_pos = {pid: i for i, pid in enumerate(top_patients)}
    df_swimmer['y'] = df_swimmer['patient_id'].map(y_pos)
    
    palette_resp = {'CR': '#2ECC71', 'PR': '#3498DB', 'SD': '#F1C40F', 'PD': '#E74C3C', 'NE': '#95A5A6', 'Unknown': '#BDC3C7'}
    
    for pid in top_patients:
        pt_data = df_swimmer[df_swimmer['patient_id'] == pid]
        axes[1].plot(pt_data['days_from_baseline'], pt_data['y'], color='gray', alpha=0.5, zorder=1)
        
    scatter = sns.scatterplot(data=df_swimmer, x='days_from_baseline', y='y', hue='response',
                              palette=palette_resp, s=60, zorder=2, ax=axes[1])
    axes[1].set_yticks(range(len(top_patients)))
    axes[1].set_yticklabels(top_patients, fontsize=8)
    axes[1].set_title('Serial Liquid Biopsy Timelines (Top 25 Followed Patients)')
    axes[1].set_xlabel('Days from Baseline')
    axes[1].set_ylabel('Patient Identifier')
    axes[1].axvline(42, color='green', linestyle=':', label='Cycle 3 Landmark (~Day 42)')
    axes[1].legend(title='Response', loc='lower right')
    plt.tight_layout()
    plot1_p = os.path.join(output_dir, '01_biomarkers_longitudinal_timelines.png')
    plt.savefig(plot1_p)
    plt.close()

    # --- PLOT 2: Quantitative ctDNA Concentration & VAF Distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Log-transformed ctDNA values (+1 for log scale)
    log_ctdna = np.log10(df_b['ctdna_value'] + 1.0)
    sns.histplot(log_ctdna, kde=True, color='#8E44AD', bins=25, ax=axes[0])
    axes[0].set_title('ctDNA Concentration Distribution [log10(molecules/mL + 1)]')
    axes[0].set_xlabel('log10(ctDNA mutant molecules/mL + 1)')
    axes[0].set_ylabel('Observation Count')
    zero_ctdna = (df_b['ctdna_value'] == 0).sum()
    axes[0].text(0.05, 0.90, f"Undetectable / Zero ctDNA:\n{zero_ctdna} observations ({zero_ctdna/total_records*100:.1f}%)",
                 transform=axes[0].transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    # Variant Allele Frequency (VAF)
    sns.histplot(df_b['vaf'] * 100.0, kde=True, color='#D35400', bins=25, ax=axes[1])
    axes[1].set_title('Variant Allele Frequency Distribution (VAF %)')
    axes[1].set_xlabel('Mean Variant Allele Frequency (%)')
    axes[1].set_ylabel('Observation Count')
    axes[1].axvline(df_b['vaf'].median()*100.0, color='black', linestyle='--', label=f"Median: {df_b['vaf'].median()*100.0:.2f}%")
    axes[1].legend()
    plt.tight_layout()
    plot2_p = os.path.join(output_dir, '02_biomarkers_ctdna_and_vaf_distributions.png')
    plt.savefig(plot2_p)
    plt.close()

    # --- PLOT 3: PD-L1 Expression & Treatment Responses ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pdl1_valid = df_b.dropna(subset=['protein_marker'])
    sns.histplot(pdl1_valid['protein_marker'], kde=False, color='#16A085', bins=20, ax=axes[0])
    axes[0].set_title('Baseline PD-L1 MPS Expression (%) Distribution')
    axes[0].set_xlabel('PD-L1 Expression (%)')
    axes[0].set_ylabel('Count (312 valid / 315 total)')
    axes[0].axvline(pdl1_valid['protein_marker'].median(), color='black', linestyle='--',
                    label=f"Median: {pdl1_valid['protein_marker'].median():.1f}%")
    axes[0].legend()

    # RECIST 1.1 Response vs Survival Outcome
    ct_resp_out = pd.crosstab(df_b['response'], df_b['outcome'])
    ct_resp_out.plot(kind='bar', stacked=True, color=['#E74C3C', '#2ECC71', '#95A5A6'], ax=axes[1])
    axes[1].set_title('Clinical Response vs Patient Survival Status')
    axes[1].set_xlabel('RECIST 1.1 Best Overall Response')
    axes[1].set_ylabel('Number of Observations')
    axes[1].legend(title='Vital Status')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plot3_p = os.path.join(output_dir, '03_biomarkers_pdl1_response_outcome.png')
    plt.savefig(plot3_p)
    plt.close()

    # --- PLOT 4: Longitudinal Trajectories by RECIST Response Category ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    resp_categories = [('PR', 'Partial Response (Favorable)', '#3498DB'),
                       ('SD', 'Stable Disease (Intermediate)', '#F39C12'),
                       ('PD', 'Progressive Disease (Unfavorable)', '#E74C3C')]
                       
    for ax, (cat, title, colr) in zip(axes, resp_categories):
        subset = df_b[df_b['response'] == cat]
        for pid, pt_grp in subset.groupby('patient_id'):
            if len(pt_grp) > 1:
                ax.plot(pt_grp['days_from_baseline'], np.log10(pt_grp['ctdna_value'] + 1.0),
                        marker='o', markersize=3, alpha=0.4, color=colr)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Days from Baseline')
        ax.set_ylabel('log10(ctDNA molecules/mL + 1)')
        ax.set_ylim(-0.2, 4.5)
        ax.axvline(42, color='gray', linestyle=':', label='Cycle 3 (~Wk 6)')
        ax.legend(fontsize=8)
    plt.suptitle('Serial ctDNA Dynamics Classified by RECIST 1.1 Clinical Trajectory', fontsize=12)
    plt.tight_layout()
    plot4_p = os.path.join(output_dir, '04_biomarkers_serial_trajectories.png')
    plt.savefig(plot4_p)
    plt.close()

    print("Biomarker EDA completed successfully.")
    return {
        'total_records': total_records,
        'unique_patients': unique_patients,
        'chronological_violations': chrono_violations,
        'duplicate_visits': int(dup_visits),
        'median_visits_per_patient': float(rec_per_pt.median()),
        'max_visits_per_patient': int(rec_per_pt.max()),
        'missing_values': missing_summary[missing_summary > 0].to_dict(),
        'ctdna_median': float(ctdna_stats['50%']),
        'vaf_median': float(vaf_stats['50%']),
        'pdl1_median': float(pdl1_stats['50%'])
    }

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Stage 02 Comprehensive EDA Engineering")
    parser.add_argument("data_dir", nargs="?", default=None, help="Path to Stage02_Data dataset directory")
    parser.add_argument("--output_dir", default=None, help="Target directory for EDA plots and tables")
    args = parser.parse_args()

    dataset_dir = get_dataset_dir(args.data_dir)
    print(f"Target Dataset Directory: {dataset_dir}")
    if not os.path.isdir(dataset_dir):
        print(f"ERROR: Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output = args.output_dir if args.output_dir else os.path.join(script_dir, 'outputs')
    base_output = os.path.abspath(base_output)
    print(f"EDA Outputs Directory: {base_output}")
    
    histo_out, rad_out, bio_out = setup_output_dirs(base_output)

    # Execute Modality EDAs
    histo_results = run_histopathology_eda(dataset_dir, histo_out)
    rad_results = run_radiology_eda(dataset_dir, rad_out)
    bio_results = run_biomarker_eda(dataset_dir, bio_out)

    print("\n" + "="*70)
    print("ALL STAGE 02 EDA ENGINEERING OPERATIONS COMPLETED SUCCESSFULLY!")
    print(f"Histopathology outputs saved to: {histo_out}")
    print(f"Radiology outputs saved to:      {rad_out}")
    print(f"Biomarker outputs saved to:      {bio_out}")
    print("="*70)

if __name__ == '__main__':
    main()
