import numpy as np
import rasterio as rs
import json
import os
from tqdm import tqdm
import shutil
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

"""
For ROIs with exactly 2 target images:
- Calculate cloud coverage for each image using cld_shdw mask files
- Select the one with lower cloud coverage
"""

# Configuration for cloud coverage calculation
CLOUD_THRESHOLD = 50  # Pixels with cloud probability > 50% are considered cloudy


def get_cloud_mask_path(s2_toa_path):
    """
    Convert s2_toa path to corresponding cld_shdw path.
    Example: roi100488/2022_1/s2_toa/roi100488_s2_toa_2022_1_1_median.tif
          -> roi100488/2022_1/cld_shdw/roi100488_cld_shdw_2022_1_1_median.tif
    """
    cloud_path = s2_toa_path.replace('/s2_toa/', '/cld_shdw/')
    cloud_path = cloud_path.replace('_s2_toa_', '_cld_shdw_')
    return cloud_path


def calculate_cloud_coverage_from_mask(mask_path, threshold=CLOUD_THRESHOLD):
    """
    Calculate cloud coverage percentage from cloud probability mask.
    
    The cld_shdw files have 5 bands with values 0-100 representing cloud probability.
    We use band 1 (primary cloud probability) and consider pixels > threshold as cloudy.
    
    Returns:
        float: cloud coverage percentage, or 100.0 if error
    """
    try:
        if not os.path.exists(mask_path):
            return 100.0
            
        with rs.open(mask_path) as src:
            # Read band 1 (cloud probability)
            cloud_prob = src.read(1)
            
            # Total valid pixels (exclude any potential nodata)
            valid_mask = ~np.isnan(cloud_prob)
            total_pixels = np.sum(valid_mask)
            
            if total_pixels == 0:
                return 100.0
            
            # Pixels considered cloudy (probability > threshold)
            cloudy_pixels = np.sum((cloud_prob > threshold) & valid_mask)
            cloud_coverage = (cloudy_pixels / total_pixels) * 100
            
            return cloud_coverage
            
    except Exception as e:
        print(f"Error processing {mask_path}: {e}")
        return 100.0  # Assume worst case


def process_single_roi(args):
    """
    For a ROI with 2 target images:
    1. Calculate cloud coverage for both using cld_shdw masks
    2. Return the relative path of the better one
    """
    roi_id, image_paths, data_root = args
    
    try:
        # Calculate cloud coverage for both images using cloud masks
        coverages = []
        for path in image_paths:
            # Get the cloud mask path for this s2_toa image
            mask_rel_path = get_cloud_mask_path(path)
            mask_full_path = os.path.join(data_root, mask_rel_path)
            
            if not os.path.exists(mask_full_path):
                return (roi_id, False, f"Cloud mask not found: {mask_rel_path}", None)
            
            coverage = calculate_cloud_coverage_from_mask(mask_full_path)
            coverages.append(coverage)
        
        # Select image with lower cloud coverage
        best_idx = np.argmin(coverages)
        best_path = image_paths[best_idx]
        best_coverage = coverages[best_idx]
        
        return (roi_id, True, None, {
            'image1_coverage': coverages[0],
            'image2_coverage': coverages[1],
            'selected': best_idx + 1,
            'best_coverage': best_coverage,
            'best_path': best_path
        })
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return (roi_id, False, error_msg, None)


def main():
    # Configuration
    json_file = '/home/wangyu/final_CCR/Filter/300km/combined_coastline_and_sea_300km.json'
    data_root = '/mnt/sdb/jingfeng/data'
    output_dir = '/home/wangyu/final_CCR/low_coverage'
    output_file = os.path.join(output_dir, 'best_target_paths.json')
    n_workers = 15
    
    print("="*80)
    print("SELECT BEST IMAGE FROM 2-TARGET ROIs")
    print("="*80)
    print(f"Strategy: Pick image with lower cloud coverage")
    print(f"Output: {output_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load JSON
    print(f"\n[1] Loading JSON...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Group by ROI and find ROIs with exactly 2 targets
    roi_targets = {}
    for key, value in tqdm(data.items(), desc="Scanning"):
        roi_id = value['roi'][0]
        targets = value.get('target', [])
        
        if roi_id not in roi_targets:
            roi_targets[roi_id] = []
        
        for target in targets:
            if len(target) > 1:
                rel_path = target[1]
                if rel_path not in roi_targets[roi_id]:
                    roi_targets[roi_id].append(rel_path)
    
    # Filter for exactly 2 targets
    two_target_rois = {
        roi_id: paths for roi_id, paths in roi_targets.items()
        if len(paths) == 2
    }
    
    print(f"\n[2] Found {len(two_target_rois)} ROIs with exactly 2 target images")
    
    if len(two_target_rois) == 0:
        print("No 2-target ROIs found!")
        return
    
    # Prepare processing arguments
    process_args = [
        (roi_id, paths, data_root)
        for roi_id, paths in two_target_rois.items()
    ]

    
    print(f"\n[3] Processing with {n_workers} workers...")
    
    success_count = 0
    failed_rois = []
    all_stats = []
    
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_roi, process_args),
            total=len(process_args),
            desc="Selecting best images"
        ))
    
    # Collect results and build ROI dictionary
    roi_paths_dict = {}
    for roi_id, success, error, stats in results:
        if success:
            success_count += 1
            all_stats.append(stats)
            roi_paths_dict[roi_id] = stats['best_path']
        else:
            failed_rois.append((roi_id, error))
    
    # Write output JSON file with ROI numbers as keys
    print(f"\n[4] Writing output file...")
    output_data = {
        "metadata": {
            "description": "Best target image paths (lower cloud coverage)",
            "total_rois": success_count,
            "failed_rois": len(failed_rois)
        },
        "roi_paths": roi_paths_dict
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Write log
    log_file = os.path.join(output_dir, 'selection_log.txt')
    with open(log_file, 'w') as log:
        log.write(f"Best Image Selection from 2-Target ROIs\n")
        log.write(f"="*80 + "\n\n")
        
        log.write(f"Results:\n")
        log.write(f"  Total: {len(two_target_rois)}\n")
        log.write(f"  Success: {success_count}\n")
        log.write(f"  Failed: {len(failed_rois)}\n\n")
        
        if all_stats:
            avg_coverage = np.mean([s['best_coverage'] for s in all_stats])
            log.write(f"  Average selected coverage: {avg_coverage:.2f}%\n")
            
            # Distribution of which image was selected
            selected_1 = sum(1 for s in all_stats if s['selected'] == 1)
            selected_2 = sum(1 for s in all_stats if s['selected'] == 2)
            log.write(f"  Image 1 selected: {selected_1}\n")
            log.write(f"  Image 2 selected: {selected_2}\n\n")
        
        if failed_rois:
            log.write(f"Failed ROIs:\n")
            for roi_id, error in failed_rois:
                log.write(f"  {roi_id}: {error}\n")
    
    # Summary
    print(f"\n" + "="*80)
    print("SELECTION COMPLETE")
    print("="*80)
    print(f"\n✅ Success: {success_count}/{len(two_target_rois)}")
    print(f"❌ Failed: {len(failed_rois)}")
    
    if all_stats:
        avg_coverage = np.mean([s['best_coverage'] for s in all_stats])
        print(f"\n📊 Average cloud coverage: {avg_coverage:.2f}%")
        
        selected_1 = sum(1 for s in all_stats if s['selected'] == 1)
        selected_2 = sum(1 for s in all_stats if s['selected'] == 2)
        print(f"   Image 1 selected: {selected_1} times")
        print(f"   Image 2 selected: {selected_2} times")
    
    print(f"\n📂 Output: {output_file}")
    print(f"   Contains: {success_count} relative paths")
    print("="*80)


if __name__ == "__main__":
    main()
