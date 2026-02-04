import numpy as np
import rasterio as rs
from tqdm import tqdm
import json
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

"""
Low-Rank Decomposition for Satellite Image Compositing (PARALLEL VERSION)
========================================================================
Creates optimal composite target images from multiple low-cloud (<10%) satellite images
using Robust PCA to separate clean background (low-rank) from residual noise/clouds (sparse).

The algorithm:
1. Stacks multiple target images (with <10% cloud coverage) into a matrix
2. Applies Robust PCA: X = L + S where:
   - L (low-rank): Clean, consistent background features
   - S (sparse): Residual clouds, sensor noise, temporal changes
3. Creates final composite by taking median of low-rank components
"""

def load_multiband_image(image_path):
    """Load a multi-band satellite image with error handling."""
    try:
        with rs.open(image_path) as src:
            data = src.read()
            meta = src.meta.copy()
            # Mask invalid values
            data = np.ma.masked_invalid(data)
            data = np.ma.filled(data, 0)
        return data, meta
    except Exception as e:
        raise IOError(f"Failed to load {image_path}: {str(e)}")


def normalize_band(band_data):
    """Normalize band to [0, 1] range for better RPCA performance."""
    band_min = np.percentile(band_data, 2)
    band_max = np.percentile(band_data, 98)
    if band_max > band_min:
        normalized = (band_data - band_min) / (band_max - band_min)
        return np.clip(normalized, 0, 1), band_min, band_max
    return band_data, band_min, band_max


def denormalize_band(normalized_data, band_min, band_max):
    """Restore original intensity range."""
    return normalized_data * (band_max - band_min) + band_min


def construct_image_matrix_per_band(image_paths, band_idx=0):
    """
    Construct matrix X for a specific band where each column is a flattened image.
    Also returns normalization parameters for denormalization.
    """
    first_data, _ = load_multiband_image(image_paths[0])
    n_pixels = first_data.shape[1] * first_data.shape[2]
    n_images = len(image_paths)
    X = np.zeros((n_pixels, n_images))
    norm_params = []
    
    for i, path in enumerate(image_paths):
        data, _ = load_multiband_image(path)
        band_data = data[band_idx]
        normalized, band_min, band_max = normalize_band(band_data.flatten())
        X[:, i] = normalized
        norm_params.append((band_min, band_max))
    
    return X, norm_params


def soft_threshold(X, threshold):
    """Soft thresholding operator for sparse component."""
    return np.sign(X) * np.maximum(np.abs(X) - threshold, 0)


def robust_pca(X, lambda_param=None, mu=None, max_iter=100, tol=1e-7, verbose=False):
    """
    Robust PCA using Principal Component Pursuit algorithm.
    
    Decomposes X = L + S where:
    - L is low-rank (clean background)
    - S is sparse (noise, clouds, outliers)
    
    Parameters:
    -----------
    X : array (m, n)
        Input matrix where each column is a flattened image
    lambda_param : float
        Sparsity parameter (default: 1/sqrt(max(m,n)))
    mu : float
        Step size parameter (default: 0.25/mean(|X|))
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    
    Returns:
    --------
    L : array
        Low-rank component (clean composite)
    S : array
        Sparse component (anomalies/clouds)
    """
    m, n = X.shape
    
    # Default parameters optimized for satellite imagery
    if lambda_param is None:
        lambda_param = 1.0 / np.sqrt(max(m, n))
    if mu is None:
        mu = 0.25 / (np.abs(X).mean() + 1e-8)
    
    # Initialize
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    
    for iteration in range(max_iter):
        # Update L via SVD soft-thresholding
        U, sigma, Vt = np.linalg.svd(X - S + Y/mu, full_matrices=False)
        sigma_thresh = np.maximum(sigma - 1/mu, 0)
        L = U @ np.diag(sigma_thresh) @ Vt
        
        # Update S via element-wise soft-thresholding
        S = soft_threshold(X - L + Y/mu, lambda_param/mu)
        
        # Update dual variable Y
        Y = Y + mu * (X - L - S)
        
        # Check convergence
        error = np.linalg.norm(X - L - S, 'fro') / (np.linalg.norm(X, 'fro') + 1e-8)
        
        if verbose and iteration % 10 == 0:
            print(f"    Iteration {iteration}: Error = {error:.6f}")
        
        if error < tol:
            if verbose:
                print(f"    Converged at iteration {iteration}")
            break
    
    return L, S


def create_optimal_composite(image_paths, lambda_param=0.01, max_iter=100):
    """
    Create optimal composite from multiple target images using RPCA.
    
    Parameters:
    -----------
    image_paths : list
        Paths to target images (all with <10% cloud coverage)
    lambda_param : float
        RPCA sparsity parameter (lower = more aggressive cloud removal)
    max_iter : int
        Maximum RPCA iterations
    
    Returns:
    --------
    composite : array (bands, height, width)
        Optimal composite image
    metadata : dict
        Rasterio metadata
    stats : dict
        Processing statistics
    """
    # Load first image for dimensions and metadata
    first_data, metadata = load_multiband_image(image_paths[0])
    n_bands = first_data.shape[0]
    img_height, img_width = first_data.shape[1], first_data.shape[2]
    n_images = len(image_paths)
    
    # Store low-rank components for all bands
    L_all_bands = np.zeros((n_bands, img_height * img_width, n_images))
    
    # Process each band independently
    for band_idx in range(n_bands):
        # Construct image matrix (normalized)
        X, norm_params = construct_image_matrix_per_band(image_paths, band_idx)
        
        # Apply Robust PCA
        L, S = robust_pca(X, lambda_param=lambda_param, max_iter=max_iter)
        
        # Denormalize each column
        for i in range(n_images):
            band_min, band_max = norm_params[i]
            L[:, i] = denormalize_band(L[:, i], band_min, band_max)
        
        L_all_bands[band_idx] = L
    
    # Create final composite using median of low-rank components
    # Median is robust to remaining outliers
    composite = np.zeros((n_bands, img_height, img_width))
    for band_idx in range(n_bands):
        composite_band = np.median(L_all_bands[band_idx], axis=1)
        composite[band_idx] = composite_band.reshape((img_height, img_width))
    
    # Calculate statistics
    stats = {
        'n_images_used': n_images,
        'n_bands': n_bands,
        'image_shape': (img_height, img_width),
        'lambda_param': lambda_param,
        'max_iter': max_iter
    }
    
    return composite, metadata, stats


def save_multiband_tif(image, output_path, metadata):
    """Save multi-band image as GeoTIFF with proper data type."""
    meta = metadata.copy()
    meta.update({
        'dtype': 'float32',
        'count': image.shape[0],
        'compress': 'lzw'  # Add compression
    })
    
    with rs.open(output_path, 'w', **meta) as dst:
        for band_idx in range(image.shape[0]):
            dst.write(image[band_idx].astype(np.float32), band_idx + 1)


def process_single_roi(args):
    """
    Process a single ROI: create optimal composite from multiple target images.
    """
    roi_id, image_paths, output_dir, lambda_param, max_iter = args
    
    try:
        # Create optimal composite
        composite, metadata, stats = create_optimal_composite(
            image_paths, 
            lambda_param=lambda_param,
            max_iter=max_iter
        )
        
        # Save composite
        output_path = os.path.join(output_dir, f'{roi_id}_optimal_composite.tif')
        save_multiband_tif(composite, output_path, metadata)
        
        return (roi_id, True, None, stats)
    
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        return (roi_id, False, error_msg, None)


def main():
    # ===== Configuration =====
    json_file = '/home/wangyu/final_CCR/Filter/300km/coastline_sea_summary_300km.json'
    data_root = '/mnt/sdb/jingfeng/data'
    output_dir = '/home/wangyu/final_CCR/optimal_composites'
    
    # Processing parameters
    min_target_images = 3  # Minimum images needed for meaningful decomposition
    lambda_param = 0.01    # RPCA sparsity (0.001-0.05: lower = more aggressive)
    max_iter = 100         # RPCA iterations
    n_workers = 15  # Use 15 worker cores
    
    print("="*80)
    print("LOW-RANK DECOMPOSITION FOR OPTIMAL SATELLITE IMAGE COMPOSITING")
    print("="*80)
    print(f"\nPurpose:")
    print(f"  Create optimal composite 'ground truth' images from multiple low-cloud")
    print(f"  target images using Robust PCA to separate clean signal from noise.")
    
    print(f"\nConfiguration:")
    print(f"  Minimum target images: {min_target_images}")
    print(f"  RPCA lambda (sparsity): {lambda_param}")
    print(f"  Max RPCA iterations: {max_iter}")
    print(f"  Output directory: {output_dir}")
    print(f"  Parallel workers: {n_workers}/{cpu_count()} cores")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ===== Load and filter data =====
    print(f"\n[1] Loading and analyzing JSON...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Group target images by ROI
    roi_images = {}
    for key, value in tqdm(data.items(), desc="Scanning JSON"):
        roi_id = value['roi'][0]
        targets = value.get('target', [])
        
        if roi_id not in roi_images:
            roi_images[roi_id] = []
        
        for target in targets:
            if len(target) > 1:
                rel_path = target[1]
                full_path = os.path.join(data_root, rel_path)
                if full_path not in roi_images[roi_id]:
                    roi_images[roi_id].append(full_path)
    
    # Filter ROIs with sufficient images
    qualifying_rois = {
        roi_id: paths for roi_id, paths in roi_images.items() 
        if len(paths) >= min_target_images
    }
    
    print(f"\n[2] ROI Statistics:")
    print(f"  Total unique ROIs: {len(roi_images)}")
    print(f"  ROIs with {min_target_images}+ target images: {len(qualifying_rois)}")
    
    # Distribution of image counts
    image_counts = [len(paths) for paths in qualifying_rois.values()]
    if image_counts:
        print(f"  Target images per ROI: min={min(image_counts)}, "
              f"max={max(image_counts)}, mean={np.mean(image_counts):.1f}")
    
    # Verify file existence
    print(f"\n[3] Verifying files...")
    valid_rois = {}
    for roi_id, paths in tqdm(qualifying_rois.items(), desc="Checking files"):
        existing_paths = [p for p in paths if os.path.exists(p)]
        if len(existing_paths) >= min_target_images:
            valid_rois[roi_id] = existing_paths
    
    print(f"  Valid ROIs: {len(valid_rois)}")
    
    if len(valid_rois) == 0:
        print("\n❌ No valid ROIs to process!")
        return
    
    # ===== Parallel processing =====
    process_args = [
        (roi_id, image_paths, output_dir, lambda_param, max_iter) 
        for roi_id, image_paths in valid_rois.items()
    ]
    
    print(f"\n[4] Creating optimal composites for {len(valid_rois)} ROIs...")
    print(f"    Using {n_workers} parallel workers")
    
    success_count = 0
    failed_rois = []
    all_stats = []
    
    start_time = datetime.now()
    
    with Pool(processes=n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_roi, process_args),
            total=len(process_args),
            desc="Processing ROIs"
        ))
    
    # ===== Collect results =====
    for roi_id, success, error, stats in results:
        if success:
            success_count += 1
            all_stats.append(stats)
        else:
            failed_rois.append((roi_id, error))
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    # ===== Write log =====
    log_file = os.path.join(output_dir, 'processing_log.txt')
    with open(log_file, 'w') as log:
        log.write(f"Low-Rank Decomposition Processing Log\n")
        log.write(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Elapsed: {elapsed}\n")
        log.write(f"="*80 + "\n\n")
        
        log.write(f"Configuration:\n")
        log.write(f"  Lambda: {lambda_param}\n")
        log.write(f"  Max iterations: {max_iter}\n")
        log.write(f"  Min images: {min_target_images}\n")
        log.write(f"  Workers: {n_workers}\n\n")
        
        log.write(f"Results:\n")
        log.write(f"  Total: {len(valid_rois)}\n")
        log.write(f"  Success: {success_count}\n")
        log.write(f"  Failed: {len(failed_rois)}\n\n")
        
        if all_stats:
            avg_images = np.mean([s['n_images_used'] for s in all_stats])
            log.write(f"  Average images per composite: {avg_images:.1f}\n")
        
        if failed_rois:
            log.write(f"\nFailed ROIs:\n")
            for roi_id, error in failed_rois:
                log.write(f"  {roi_id}: {error}\n")
    
    # ===== Summary =====
    print(f"\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    
    elapsed_seconds = elapsed.total_seconds()
    hours = int(elapsed_seconds // 3600)
    minutes = int((elapsed_seconds % 3600) // 60)
    seconds = int(elapsed_seconds % 60)
    
    print(f"\n⏱️  Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    if len(valid_rois) > 0:
        print(f"  Average: {elapsed_seconds/len(valid_rois):.1f}s per ROI")
    
    print(f"\n📊 Results:")
    print(f"  ✅ Success: {success_count}/{len(valid_rois)}")
    print(f"  ❌ Failed: {len(failed_rois)}")
    
    if all_stats:
        avg_images = np.mean([s['n_images_used'] for s in all_stats])
        print(f"\n📸 Average images per composite: {avg_images:.1f}")
    
    if failed_rois:
        print(f"\n⚠️  First 5 failed ROIs:")
        for roi_id, error in failed_rois[:5]:
            print(f"  • {roi_id}: {error[:60]}...")
    
    print(f"\n📂 Output:")
    print(f"  Directory: {output_dir}")
    print(f"  Files: {success_count} × *_optimal_composite.tif")
    print(f"  Log: processing_log.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()