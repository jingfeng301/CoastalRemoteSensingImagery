#!/usr/bin/env python3
"""
Combined Pipeline: ROI Extraction + Input Filtering
Combines full_data.py and filter_inputs_keep_rois.py into a single workflow

STEP 1: Extract all ROIs with targets and available inputs
STEP 2: Filter out bad/corrupted input images (keep all ROIs)
"""

import json
import os
import csv as csv_module
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import rasterio
import warnings
from datetime import datetime
import sys

warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: ROI EXTRACTION (from full_data.py)
# =============================================================================

def get_cloud_mask_path(s2_toa_path):
    """Convert s2_toa path to corresponding cld_shdw path."""
    cloud_path = s2_toa_path.replace('/s2_toa/', '/cld_shdw/')
    cloud_path = cloud_path.replace('_s2_toa_', '_cld_shdw_')
    return cloud_path


def find_s2_images_with_masks(roi_dir, data_root, exclude_paths):
    """Find all s2_toa images in an ROI directory that have matching masks."""
    valid_s2_images = []
    
    for year_month_dir in roi_dir.glob('*'):
        if year_month_dir.is_dir():
            s2_toa_dir = year_month_dir / 's2_toa'
            if s2_toa_dir.exists():
                for tif_file in s2_toa_dir.glob('*.tif'):
                    s2_rel_path = str(tif_file.relative_to(data_root))
                    
                    if s2_rel_path in exclude_paths:
                        continue
                    
                    mask_rel_path = get_cloud_mask_path(s2_rel_path)
                    mask_full_path = os.path.join(data_root, mask_rel_path)
                    
                    if os.path.exists(mask_full_path):
                        valid_s2_images.append(s2_rel_path)
    
    return valid_s2_images


def process_roi_extraction(args):
    """Process a single ROI to find available input images."""
    roi_id, targets, n_targets, chosen_target, data_root, all_targets_in_json = args
    
    roi_num = roi_id.replace('roi', '')
    roi_dir = Path(data_root) / roi_id
    
    if not roi_dir.exists():
        return (roi_num, chosen_target, [], 0)
    
    # Determine what to exclude based on number of targets
    if n_targets == 1:
        exclude_paths = set([chosen_target])
    elif n_targets == 2:
        exclude_paths = set([chosen_target])
    else:  # 3+ targets
        exclude_paths = set(all_targets_in_json)
    
    valid_s2_images = find_s2_images_with_masks(roi_dir, data_root, exclude_paths)
    
    return (roi_num, chosen_target, valid_s2_images, len(valid_s2_images))


def extract_rois_target(json_file, data_root, best_paths_json, composite_dir, output_csv, num_workers=15):
    """Extract all ROIs with targets and their available input images."""
    print("\n" + "="*80)
    print("STEP 1: EXTRACTING ALL ROIs WITH TARGETS AND AVAILABLE INPUTS")
    print("="*80)
    
    print(f"\n[1.1] Loading JSON file...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    print(f"  Total entries: {len(data):,}")
    
    print(f"\n[1.2] Loading best target paths...")
    with open(best_paths_json, 'r') as f:
        best_paths_data = json.load(f)
    best_paths = best_paths_data.get('roi_paths', {})
    print(f"  Total ROIs with best paths: {len(best_paths):,}")
    
    print(f"\n[1.3] Grouping targets by ROI...")
    roi_targets = {}
    roi_all_json_targets = {}
    
    for key, value in tqdm(data.items(), desc="Processing entries"):
        roi_id = value['roi'][0]
        targets = value.get('target', [])
        
        if roi_id not in roi_targets:
            roi_targets[roi_id] = []
            roi_all_json_targets[roi_id] = []
        
        for target in targets:
            if len(target) > 1:
                rel_path = target[1]
                if rel_path not in roi_all_json_targets[roi_id]:
                    roi_all_json_targets[roi_id].append(rel_path)
                
                full_path = os.path.join(data_root, rel_path)
                if os.path.exists(full_path) and rel_path not in roi_targets[roi_id]:
                    roi_targets[roi_id].append(rel_path)
    
    print(f"\n[1.4] Preparing ROIs for processing...")
    process_args = []
    stats = {
        '1_target': 0,
        '2_targets_matched': 0,
        '2_targets_unmatched': 0,
        '3+_targets': 0,
        '3+_missing_composite': 0,
        'skipped': 0
    }
    
    for roi_id in sorted(roi_targets.keys()):
        targets = roi_targets[roi_id]
        all_json_targets = roi_all_json_targets[roi_id]
        n_targets = len(targets)
        roi_num = roi_id.replace('roi', '')
        
        if n_targets == 0:
            stats['skipped'] += 1
            continue
        
        elif n_targets == 1:
            target_full_path = os.path.join(data_root, targets[0])
            if os.path.exists(target_full_path):
                chosen_target = targets[0]
                process_args.append((roi_id, targets, n_targets, chosen_target, data_root, all_json_targets))
                stats['1_target'] += 1
            else:
                stats['skipped'] += 1
        
        elif n_targets == 2:
            if roi_id in best_paths:
                chosen_target = best_paths[roi_id]
                stats['2_targets_matched'] += 1
            else:
                chosen_target = targets[0]
                stats['2_targets_unmatched'] += 1
                print(f"  Warning: {roi_id} not found in best_paths.json, using first target")
            
            process_args.append((roi_id, targets, n_targets, chosen_target, data_root, all_json_targets))
        
        else:  # 3+ targets
            composite_path = os.path.join(composite_dir, f"roi{roi_num}_optimal_composite.tif")
            
            if os.path.exists(composite_path):
                chosen_target = os.path.basename(composite_path)
                process_args.append((roi_id, targets, n_targets, chosen_target, data_root, all_json_targets))
                stats['3+_targets'] += 1
            else:
                stats['3+_missing_composite'] += 1
                print(f"  Warning: Missing composite for {roi_id} at {composite_path}")
    
    print(f"\n[1.5] Finding available inputs with {num_workers} workers...")
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_roi_extraction, process_args),
            total=len(process_args),
            desc="Processing ROIs"
        ))
    
    other_s2_counts = []
    max_inputs = 0
    
    for roi_num, chosen_target, valid_s2_images, other_s2_count in results:
        other_s2_counts.append(other_s2_count)
        if other_s2_count > max_inputs:
            max_inputs = other_s2_count
    
    print(f"\n  Maximum number of inputs per ROI: {max_inputs}")
    
    csv_rows = []
    for roi_num, chosen_target, valid_s2_images, other_s2_count in results:
        row = {
            'roi_number': roi_num,
            'target_path': chosen_target
        }
        
        for i, s2_image in enumerate(valid_s2_images, start=1):
            row[f'input_{i}'] = s2_image
        
        for i in range(len(valid_s2_images) + 1, max_inputs + 1):
            row[f'input_{i}'] = ''
        
        csv_rows.append(row)
    
    csv_rows.sort(key=lambda x: int(x['roi_number']))
    
    print(f"\n[1.6] Writing CSV to {output_csv}...")
    fieldnames = ['roi_number', 'target_path'] + [f'input_{i}' for i in range(1, max_inputs + 1)]
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"\n[1.7] Extraction Statistics:")
    print(f"  Total ROIs: {len(roi_targets):,}")
    print(f"  ROIs with exactly 1 target: {stats['1_target']:,}")
    print(f"  ROIs with exactly 2 targets: {stats['2_targets_matched'] + stats['2_targets_unmatched']:,}")
    print(f"  ROIs with 3+ targets: {stats['3+_targets']:,}")
    print(f"  Total rows written: {len(csv_rows):,}")
    
    if other_s2_counts:
        print(f"\n[1.8] Input Images Statistics:")
        print(f"  Min inputs per ROI: {min(other_s2_counts)}")
        print(f"  Max inputs per ROI: {max(other_s2_counts)}")
        print(f"  Mean inputs per ROI: {sum(other_s2_counts)/len(other_s2_counts):.1f}")
        print(f"  Total input images: {sum(other_s2_counts):,}")
    
    print(f"\n✓ STEP 1 COMPLETE - Raw dataset created")
    return output_csv


# =============================================================================
# STEP 2: INPUT FILTERING (from filter_inputs_keep_rois.py)
# =============================================================================

def calculate_nan_percentage(file_path):
    """Calculate the percentage of NaN values in a TIFF file."""
    if not os.path.exists(file_path):
        return None
    
    try:
        with rasterio.open(file_path) as src:
            img = src.read()
            is_nan = np.isnan(img)
            is_inf = np.isinf(img)
            total_pixels = img.size
            invalid_pixels = is_nan.sum() + is_inf.sum()
            nan_percentage = (invalid_pixels / total_pixels) * 100
            return nan_percentage
    except Exception as e:
        return None


def check_file_pair_nan(s2_path, data_root, nan_threshold=10.0):
    """Check if both S2 and corresponding mask file meet NaN threshold."""
    result = {
        's2_exists': False,
        'mask_exists': False,
        's2_nan_pct': np.nan,
        'mask_nan_pct': np.nan,
        'keep': False,
        'reason': ''
    }
    
    if not s2_path.startswith(data_root) and not s2_path.startswith('/'):
        s2_full_path = os.path.join(data_root, s2_path)
    else:
        s2_full_path = s2_path
    
    mask_full_path = get_cloud_mask_path(s2_full_path)
    
    result['s2_exists'] = os.path.exists(s2_full_path)
    result['mask_exists'] = os.path.exists(mask_full_path)
    
    if not result['s2_exists']:
        result['reason'] = 'S2 missing'
        return result
    
    if not result['mask_exists']:
        result['reason'] = 'mask missing'
        return result
    
    result['s2_nan_pct'] = calculate_nan_percentage(s2_full_path)
    result['mask_nan_pct'] = calculate_nan_percentage(mask_full_path)
    
    if result['s2_nan_pct'] is None:
        result['reason'] = 'S2 unreadable/corrupted'
        return result
    
    if result['mask_nan_pct'] is None:
        result['reason'] = 'mask unreadable/corrupted'
        return result
    
    if result['s2_nan_pct'] >= nan_threshold:
        result['reason'] = f'S2 NaN {result["s2_nan_pct"]:.1f}%'
        return result
    
    if result['mask_nan_pct'] >= nan_threshold:
        result['reason'] = f'mask NaN {result["mask_nan_pct"]:.1f}%'
        return result
    
    result['keep'] = True
    return result


def process_single_roi_filter(args):
    """Process a single ROI and filter its inputs."""
    row, input_cols, non_input_cols, data_root, nan_threshold = args
    
    roi_number = row['roi_number']
    
    new_row = {col: row[col] for col in non_input_cols}
    
    valid_inputs = []
    removed_inputs = []
    
    for input_col in input_cols:
        if pd.notna(row[input_col]) and str(row[input_col]).strip() != '':
            s2_path = row[input_col]
            
            check_result = check_file_pair_nan(s2_path, data_root, nan_threshold)
            
            if check_result['keep']:
                valid_inputs.append(s2_path)
            else:
                removed_inputs.append({
                    'input_col': input_col,
                    'path': s2_path,
                    'reason': check_result['reason']
                })
    
    for i, valid_path in enumerate(valid_inputs, start=1):
        new_row[f'input_{i}'] = valid_path
    
    stats = {
        'original_index': row.name,
        'roi_number': roi_number,
        'original_input_count': len([col for col in input_cols if pd.notna(row[col]) and str(row[col]).strip() != '']),
        'valid_input_count': len(valid_inputs),
        'removed_input_count': len(removed_inputs),
        'removed_details': '; '.join([f"{r['input_col']}: {r['reason']}" for r in removed_inputs])
    }
    
    return new_row, stats


def filter_inputs_keep_rois(csv_path, data_root, output_csv_path, nan_threshold=10.0, num_workers=20):
    """Filter bad input images but keep all ROIs, reorganize input columns."""
    
    print("\n" + "="*80)
    print("STEP 2: FILTERING BAD INPUT IMAGES (KEEP ALL ROIs)")
    print("="*80)
    
    print(f"\n[2.1] Configuration:")
    print(f"  Input CSV: {csv_path}")
    print(f"  Data Root: {data_root}")
    print(f"  NaN Threshold: {nan_threshold}%")
    print(f"  CPU Cores: {num_workers}")
    
    print(f"\n[2.2] Reading CSV...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df):,} entries")
    
    input_cols = [col for col in df.columns if col.startswith('input_')]
    non_input_cols = [col for col in df.columns if not col.startswith('input_')]
    print(f"✓ Found {len(input_cols)} input columns")
    
    if len(input_cols) == 0:
        print("ERROR: No input columns found in CSV!")
        return None, None
    
    results_list = []
    filtered_rows = []
    
    print(f"\n[2.3] Processing {len(df):,} ROIs with {num_workers} cores...")
    
    process_args = [
        (row, input_cols, non_input_cols, data_root, nan_threshold)
        for idx, row in df.iterrows()
    ]
    
    with Pool(processes=num_workers) as pool:
        for new_row, stats in tqdm(
            pool.imap(process_single_roi_filter, process_args),
            total=len(df),
            desc="Filtering inputs",
            ncols=100
        ):
            filtered_rows.append(new_row)
            results_list.append(stats)
    
    df_filtered = pd.DataFrame(filtered_rows)
    
    input_cols_in_new_df = sorted([col for col in df_filtered.columns if col.startswith('input_')],
                                   key=lambda x: int(x.split('_')[1]))
    final_column_order = non_input_cols + input_cols_in_new_df
    df_filtered = df_filtered[final_column_order]
    
    results_df = pd.DataFrame(results_list)
    
    print(f"\n[2.4] Filtering Statistics:")
    total_rois = len(df)
    total_original_inputs = results_df['original_input_count'].sum()
    total_valid_inputs = results_df['valid_input_count'].sum()
    total_removed_inputs = results_df['removed_input_count'].sum()
    rois_modified = (results_df['removed_input_count'] > 0).sum()
    rois_unchanged = (results_df['removed_input_count'] == 0).sum()
    
    print(f"  Total ROIs: {total_rois:,} (all kept)")
    print(f"  ROIs with all inputs valid: {rois_unchanged:,} ({rois_unchanged/total_rois*100:.1f}%)")
    print(f"  ROIs with some inputs removed: {rois_modified:,} ({rois_modified/total_rois*100:.1f}%)")
    print(f"  Original total inputs: {total_original_inputs:,}")
    print(f"  Valid inputs kept: {total_valid_inputs:,} ({total_valid_inputs/total_original_inputs*100:.1f}%)")
    print(f"  Bad inputs removed: {total_removed_inputs:,} ({total_removed_inputs/total_original_inputs*100:.1f}%)")
    
    print(f"\n[2.5] Saving filtered dataset...")
    df_filtered.to_csv(output_csv_path, index=False)
    max_inputs = len([col for col in df_filtered.columns if col.startswith('input_')])
    print(f"✓ Filtered CSV: {output_csv_path}")
    print(f"  Contains {len(df_filtered):,} ROIs")
    print(f"  Max input columns: {max_inputs}")
    
    analysis_csv_path = output_csv_path.replace('.csv', '_analysis.csv')
    results_df.to_csv(analysis_csv_path, index=False)
    print(f"✓ Analysis CSV: {analysis_csv_path}")
    
    summary_path = output_csv_path.replace('.csv', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMBINED PIPELINE SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Step 1 Output: {csv_path}\n")
        f.write(f"Step 2 Output: {output_csv_path}\n")
        f.write(f"Data Root: {data_root}\n")
        f.write(f"NaN Threshold: {nan_threshold}%\n\n")
        
        f.write(f"{'='*80}\n")
        f.write("FILTERING RESULTS\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"ROI Statistics:\n")
        f.write(f"  Total ROIs: {total_rois:,} (100% kept)\n")
        f.write(f"  ROIs with all inputs valid: {rois_unchanged:,} ({rois_unchanged/total_rois*100:.1f}%)\n")
        f.write(f"  ROIs with some inputs removed: {rois_modified:,} ({rois_modified/total_rois*100:.1f}%)\n\n")
        
        f.write(f"Input Image Statistics:\n")
        f.write(f"  Original total inputs: {total_original_inputs:,}\n")
        f.write(f"  Valid inputs kept: {total_valid_inputs:,} ({total_valid_inputs/total_original_inputs*100:.1f}%)\n")
        f.write(f"  Bad inputs removed: {total_removed_inputs:,} ({total_removed_inputs/total_original_inputs*100:.1f}%)\n")
    
    print(f"✓ Summary report: {summary_path}")
    print(f"\n✓ STEP 2 COMPLETE - Dataset filtered and cleaned")
    
    return df_filtered, results_df


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Run the complete pipeline: extraction + filtering."""
    
    print("\n" + "="*80)
    print("COMBINED PIPELINE: ROI EXTRACTION + INPUT FILTERING")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Configuration
    json_file = '/home/wangyu/final_CCR/Filter/300km/combined_coastline_and_sea_300km.json'
    data_root = '/mnt/sdb/jingfeng/data'
    best_paths_json = '/home/wangyu/final_CCR/low_coverage/best_target_paths.json'
    composite_dir = '/home/wangyu/final_CCR/optimal_composites'
    
    # Intermediate output from Step 1
    raw_csv = '/home/wangyu/final_CCR/data/all_rois_targets.csv'
    
    # Final output from Step 2
    filtered_csv = '/home/wangyu/final_CCR/data/all_rois_targets_filtered.csv'
    
    # Step 1 settings
    extraction_workers = 15
    
    # Step 2 settings
    nan_threshold = 10.0
    filtering_workers = 20
    
    # Check available cores
    available_cores = cpu_count()
    print(f"System has {available_cores} CPU cores available\n")
    
    try:
        # STEP 1: Extract ROIs and create raw dataset
        extract_rois_target(
            json_file=json_file,
            data_root=data_root,
            best_paths_json=best_paths_json,
            composite_dir=composite_dir,
            output_csv=raw_csv,
            num_workers=extraction_workers
        )
        
        # STEP 2: Filter bad inputs from the raw dataset
        filtered_df, results_df = filter_inputs_keep_rois(
            csv_path=raw_csv,
            data_root=data_root,
            output_csv_path=filtered_csv,
            nan_threshold=nan_threshold,
            num_workers=filtering_workers
        )
        
        # Final summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"Outputs:")
        print(f"  1. Raw dataset: {raw_csv}")
        print(f"  2. Filtered dataset: {filtered_csv}")
        print(f"  3. Analysis report: {filtered_csv.replace('.csv', '_analysis.csv')}")
        print(f"  4. Summary report: {filtered_csv.replace('.csv', '_summary.txt')}")
        print(f"\n✓ All {len(filtered_df):,} ROIs ready for use!\n")
        
    except Exception as e:
        print(f"\nERROR during pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
