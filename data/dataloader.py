import os
import json
import glob
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted

import rasterio
from torch.utils.data import Dataset


def read_tif(path_IMG):
    """Read a GeoTIFF file"""
    return rasterio.open(path_IMG)


def read_img(tif):
    """Read image data from opened tif and handle NaN/nodata values"""
    img = tif.read().astype(np.float32)
    
    # Check for NaN values and report
    has_nan = np.isnan(img).any()
    has_inf = np.isinf(img).any()
    
    if has_nan or has_inf:
        nan_pct = np.isnan(img).sum() / img.size * 100
        inf_pct = np.isinf(img).sum() / img.size * 100
        warnings.warn(f"File {tif.name} contains:")
        if has_nan:
            warnings.warn(f"  - NaN values: {nan_pct:.2f}% of pixels")
        if has_inf:
            warnings.warn(f"  - Inf values: {inf_pct:.2f}% of pixels")
    
    # Replace NaN and inf values with 0
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Also handle nodata values if specified in the file
    if tif.nodata is not None:
        img[img == tif.nodata] = 0.0
    
    return img


def get_cloud_mask_path(s2_toa_path, data_root):
    """
    Convert s2_toa path to corresponding cld_shdw path.
    Example: roi100488/2022_1/s2_toa/roi100488_s2_toa_2022_1_1_median.tif
          -> roi100488/2022_1/cld_shdw/roi100488_cld_shdw_2022_1_1_median.tif
    """
    cloud_path = s2_toa_path.replace('/s2_toa/', '/cld_shdw/')
    cloud_path = cloud_path.replace('_s2_toa_', '_cld_shdw_')
    return os.path.join(data_root, cloud_path)


class CoastlineCloudRemovalDataset(Dataset):
    """
    DataLoader for Coastline Cloud Removal dataset
    
    This dataset loads from a CSV manifest with the following structure:
    - roi_number: ROI identifier
    - target_path: Path to target/clean image (relative or filename for composites)
    - input_1, input_2, ..., input_N: Paths to input cloudy images
    
    Data structure:
    data_root/
    ├── roi117038/
    │   ├── 2022_1/
    │   │   ├── s2_toa/           ← Sentinel-2 images (cloudy)
    │   │   └── cld_shdw/         ← Cloud/shadow probability masks (0-100%)
    │   └── ...
    └── composite_dir/            ← Optimal composites for 3+ targets
        └── roi263_optimal_composite.tif
    
    Args:
        csv_path (str): Path to CSV manifest file
        data_root (str): Path to data root directory
        target_dir (str): Path to directory containing all target TIF files
        split (str): 'train', 'val', 'test', or 'all'
        n_input_samples (int): Number of input images to sample (None = use all)
        sampler (str): 'fixed', 'random', or 'all'
        return_masks (bool): Whether to return cloud/shadow masks
        return_paths (bool): Whether to return file paths
        train_ratio (float): Ratio for training split (used if no JSON provided)
        val_ratio (float): Ratio for validation split (used if no JSON provided)
        test_ratio (float): Ratio for test split (used if no JSON provided)
        random_seed (int): Random seed for reproducibility
        split_json_path (str): Path to JSON file with predefined splits
    """
    
    def __init__(
        self,
        csv_path,
        data_root,
        target_dir,
        split="train",
        n_input_samples=None,
        sampler='all',
        return_masks=True,
        return_paths=True,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        split_json_path=None
    ):
        self.csv_path = csv_path
        self.data_root = data_root
        self.target_dir = target_dir
        self.split = split
        self.n_input_t = n_input_samples
        self.sampler = sampler
        self.return_masks = return_masks
        self.return_paths = return_paths
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.split_json_path = split_json_path
        
        # Validate inputs
        assert split in ['train', 'val', 'test', 'all'], \
            "split must be one of: train, val, test, all"
        assert sampler in ['fixed', 'random', 'all'], \
            "sampler must be: fixed, random, or all"
        
        # Load CSV manifest
        self.df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(self.df)} ROIs")
        
        # Create splits
        self.roi_splits = self.create_splits()
        
        # Get samples for this split
        self.samples = self.get_samples()
        self.n_samples = len(self.samples)
        
        if self.n_samples == 0:
            self.throw_warn()
        else:
            print(f"\nLoaded {self.n_samples} samples for {split} split")
    
    def create_splits(self):
        """Create train/val/test splits from ROI numbers"""
        
        # If JSON split file is provided, use it
        if self.split_json_path and os.path.exists(self.split_json_path):
            print(f"Loading splits from JSON file: {self.split_json_path}")
            with open(self.split_json_path, 'r') as f:
                split_data = json.load(f)
            
            def extract_roi_numbers(roi_list):
                """Extract ROI numbers from various formats"""
                extracted = []
                for item in roi_list:
                    if isinstance(item, str):
                        # Handle "roiXXXXX" format
                        if item.lower().startswith('roi'):
                            # Remove "roi" prefix and any non-numeric characters
                            roi_num = ''.join(filter(str.isdigit, item))
                            if roi_num:  # Only add if we found digits
                                extracted.append(roi_num)
                        else:
                            # Assume it's already a number string
                            extracted.append(item)
                    elif isinstance(item, (int, float)):
                        # Convert numeric ROI to string
                        extracted.append(str(int(item)))
                return extracted
            
            # Initialize splits
            splits = {'train': [], 'val': [], 'test': []}
            
            # Extract ROI numbers for each split
            for split_key in ['train', 'val', 'test']:
                if split_key in split_data and split_data[split_key]:
                    splits[split_key] = extract_roi_numbers(split_data[split_key])
            
            # Handle case where we only have train/test and need val
            if not splits['val'] and splits['test']:
                print("No validation split found in JSON. Creating validation split from test set...")
                np.random.seed(self.random_seed)
                test_rois = splits['test'].copy()
                np.random.shuffle(test_rois)
                
                n_test = len(test_rois)
                # Use the val_ratio parameter to split test into val/test
                n_val = int(n_test * self.val_ratio / (self.val_ratio + self.test_ratio))
                
                splits['val'] = test_rois[:n_val]
                splits['test'] = test_rois[n_val:]
            
            # Add 'all' split (all ROIs across all splits)
            all_rois = []
            for split_key in ['train', 'val', 'test']:
                all_rois.extend(splits[split_key])
            splits['all'] = list(set(all_rois))
            
            # Print summary
            print(f"\nJSON splits loaded:")
            print(f"  Train: {len(splits['train'])} ROIs")
            print(f"  Val:   {len(splits['val'])} ROIs")
            print(f"  Test:  {len(splits['test'])} ROIs")
            print(f"  All:   {len(splits['all'])} ROIs")
            
            # Print metadata if available
            if 'metadata' in split_data:
                print(f"\nJSON metadata:")
                for key, value in split_data['metadata'].items():
                    print(f"  {key}: {value}")
            
            # Validate that all ROIs in splits exist in CSV
            csv_rois = set(self.df['roi_number'].astype(str).tolist())
            for split_key in ['train', 'val', 'test']:
                split_rois = set(splits[split_key])
                missing = split_rois - csv_rois
                if missing:
                    warnings.warn(f"Split '{split_key}' contains {len(missing)} ROIs not found in CSV")
                    if len(missing) <= 5:  # Print first few missing
                        print(f"    Missing ROIs: {list(missing)[:5]}...")
            
            return splits
        
        # Otherwise, create splits automatically
        else:
            print("No split JSON provided, creating automatic splits...")
            assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
                "train_ratio + val_ratio + test_ratio must equal 1.0"
            
            all_rois = self.df['roi_number'].astype(str).tolist()
            n_total = len(all_rois)
            
            # Set random seed for reproducibility
            np.random.seed(self.random_seed)
            
            # Shuffle ROIs
            shuffled_rois = np.array(all_rois.copy())
            np.random.shuffle(shuffled_rois)
            
            # Calculate split indices
            n_train = int(n_total * self.train_ratio)
            n_val = int(n_total * self.val_ratio)
            
            train_rois = shuffled_rois[:n_train].tolist()
            val_rois = shuffled_rois[n_train:n_train+n_val].tolist()
            test_rois = shuffled_rois[n_train+n_val:].tolist()
            
            splits = {
                'train': train_rois,
                'val': val_rois,
                'test': test_rois,
                'all': all_rois
            }
            
            print(f"\nAutomatic split created (seed={self.random_seed}):")
            print(f"  Train: {len(train_rois)} ROIs")
            print(f"  Val:   {len(val_rois)} ROIs")
            print(f"  Test:  {len(test_rois)} ROIs")
            
            return splits
    
    def get_samples(self):
        """Get samples for the current split"""
        samples = []
        
        # Filter dataframe by split
        split_rois = set(self.roi_splits[self.split])
        df_split = self.df[self.df['roi_number'].astype(str).isin(split_rois)]
        
        print(f"\nProcessing {len(df_split)} ROIs for {self.split} split...")
        
        for idx, row in tqdm(df_split.iterrows(), total=len(df_split), desc="Indexing samples"):
            roi_number = str(row['roi_number'])
            target_path = row['target_path']
            
            # Collect all input paths from input_1, input_2, ... columns
            input_paths = []
            col_idx = 1
            while f'input_{col_idx}' in df_split.columns:
                input_path = row[f'input_{col_idx}']
                if pd.notna(input_path) and input_path != '':
                    input_paths.append(input_path)
                col_idx += 1
            
            # Skip if no inputs
            if len(input_paths) == 0:
                warnings.warn(f"ROI {roi_number} has no input images, skipping")
                continue
            
            # Determine full target path
            # All targets are now in the target_dir folder
            # Extract just the filename from the target_path
            target_filename = os.path.basename(target_path)
            target_full_path = os.path.join(self.target_dir, target_filename)
            
            # Check if target exists
            if not os.path.exists(target_full_path):
                warnings.warn(f"Target not found for ROI {roi_number}: {target_full_path}, skipping")
                continue
            
            # Convert input paths to full paths and check existence
            valid_inputs = []
            for input_path in input_paths:
                input_full_path = os.path.join(self.data_root, input_path)
                if os.path.exists(input_full_path):
                    valid_inputs.append(input_path)
                else:
                    warnings.warn(f"Input not found: {input_full_path}")
            
            if len(valid_inputs) == 0:
                warnings.warn(f"ROI {roi_number} has no valid input images, skipping")
                continue
            
            sample = {
                'roi_number': roi_number,
                'roi_name': f'roi{roi_number}',
                'target_path': target_full_path,
                'input_paths': valid_inputs,  # Relative paths
            }
            samples.append(sample)
        
        return samples
    
    def __getitem__(self, idx):
        """
        Get a sample
        
        Returns:
            dict: {
                'input': {
                    'S2': [T, C, H, W],          # Time series of S2 images
                    'masks': [T, H, W],          # Cloud probability masks (0-100%)
                    'coverage': [T],              # Cloud coverage per time (0-1)
                    'indices': [T],               # Which time points were sampled
                    'paths': [str],               # File paths (if return_paths=True)
                    'roi_name': str
                },
                'target': {
                    'S2': [C, H, W],              # Target/clean image
                    'path': str,
                    'coord': [xmin, ymin, xmax, ymax]
                }
            }
        """
        sample_info = self.samples[idx]
        
        # Determine which inputs to load
        n_available = len(sample_info['input_paths'])
        
        if self.sampler == 'all' or self.n_input_t is None:
            # Use all available inputs
            indices = list(range(n_available))
        elif self.sampler == 'random':
            # Random sample
            n_to_sample = min(self.n_input_t, n_available)
            indices = sorted(np.random.choice(n_available, n_to_sample, replace=False))
        else:  # 'fixed'
            # Take first n_input_t samples
            indices = list(range(min(self.n_input_t, n_available)))
        
        # Load input S2 images
        s2_series = []
        mask_series = []
        coverage_series = []
        paths_series = []
        
        for i in indices:
            input_rel_path = sample_info['input_paths'][i]
            input_full_path = os.path.join(self.data_root, input_rel_path)
            
            try:
                # Load S2 image
                s2_tif = read_tif(input_full_path)
                s2_img = read_img(s2_tif)  # [C, H, W]
                s2_series.append(s2_img)
                paths_series.append(input_full_path)
                
                # Get coordinates from first image
                if i == indices[0]:
                    coords = list(s2_tif.bounds)
                
                s2_tif.close()
            except Exception as e:
                raise RuntimeError(f"Error loading S2 at {input_full_path}: {e}")
            
            # Load corresponding mask
            if self.return_masks:
                mask_full_path = get_cloud_mask_path(input_rel_path, self.data_root)
                
                if os.path.exists(mask_full_path):
                    try:
                        mask_tif = read_tif(mask_full_path)
                        mask_img = read_img(mask_tif)  # [C, H, W] or [H, W]
                        
                        # Handle multi-channel masks - take first channel
                        if mask_img.ndim == 3 and mask_img.shape[0] > 1:
                            mask_img = mask_img[0]
                        elif mask_img.ndim == 3:
                            mask_img = mask_img[0]
                        
                        mask_series.append(mask_img)
                        
                        # Compute cloud coverage (mask values 0-100 represent cloud probability %)
                        # Any value > 0 is considered "cloudy" for coverage calculation
                        coverage = np.mean(mask_img > 0)
                        coverage_series.append(coverage)
                        
                        mask_tif.close()
                    except Exception as e:
                        warnings.warn(f"Error loading mask at {mask_full_path}: {e}")
                        mask_series.append(np.zeros((s2_img.shape[1], s2_img.shape[2]), dtype=np.float32))
                        coverage_series.append(0.0)
                else:
                    warnings.warn(f"Mask not found: {mask_full_path}")
                    mask_series.append(np.zeros((s2_img.shape[1], s2_img.shape[2]), dtype=np.float32))
                    coverage_series.append(0.0)
        
        s2_series = np.array(s2_series)  # [T, C, H, W]
        if self.return_masks and len(mask_series) > 0:
            mask_series = np.array(mask_series)  # [T, H, W]
        
        # Load target image
        target_path = sample_info['target_path']
        try:
            target_tif = read_tif(target_path)
            target_img = read_img(target_tif)  # [C, H, W]
            target_coords = list(target_tif.bounds)
            target_tif.close()
        except Exception as e:
            raise RuntimeError(f"Error loading target at {target_path}: {e}")
        
        # Construct output dictionary
        output = {
            'input': {
                'S2': s2_series,
                'indices': indices,
                'coord': coords,
                'roi_name': sample_info['roi_name']
            },
            'target': {
                'S2': target_img,
                'coord': target_coords,
            }
        }
        
        # Add masks if requested
        if self.return_masks and len(mask_series) > 0:
            output['input']['masks'] = mask_series
            output['input']['coverage'] = coverage_series
        
        # Add paths if requested
        if self.return_paths:
            output['input']['paths'] = paths_series
            output['target']['path'] = target_path
        
        return output
    
    def __len__(self):
        return self.n_samples
    
    def throw_warn(self):
        """Warning message for missing data"""
        warnings.warn(f"""
        No valid samples found for split '{self.split}'!
        
        Expected CSV structure:
        roi_number,target_path,input_1,input_2,...
        117038,roi117038/2022_7/s2_toa/...,roi117038/2022_1/s2_toa/...,roi117038/2022_2/s2_toa/...
        
        Expected directory structure:
        {self.data_root}/
        ├── roi117038/
        │   ├── 2022_1/
        │   │   ├── s2_toa/*.tif
        │   │   └── cld_shdw/*.tif  (cloud probability masks: 0-100%)
        │   └── ...
        └── ...
        
        {self.target_dir}/
        ├── roi117038_s2_toa_2022_7_15_median.tif
        ├── roi263_optimal_composite.tif
        └── ... (all target TIF files)
        
        Checklist:
        1. CSV file exists and is readable
        2. ROI directories exist in data_root
        3. Input files exist at specified paths
        4. All target files exist in target_dir
        
        JSON split info:
        - JSON path: {self.split_json_path}
        - JSON exists: {os.path.exists(self.split_json_path) if self.split_json_path else 'Not provided'}
        - Split: {self.split}
        - Split ROIs in JSON: {len(self.roi_splits[self.split]) if hasattr(self, 'roi_splits') else 'Unknown'}
        """)


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the project root (parent of data directory)
    project_root = os.path.dirname(script_dir)
    
    print("=" * 80)
    print("Example: Using JSON splits")
    print("=" * 80)
    
    dataset_json = CoastlineCloudRemovalDataset(
        csv_path=os.path.join(script_dir, 'data.csv'),                      # Same directory as script
        data_root='/mnt/sdb/jingfeng/data',                                  # UPDATE: Path to AllClear data
        target_dir='/home/wangyu/CoastalRemoteSensingImagery/coastline_target', # UPDATE: Path to target data
        split_json_path=os.path.join(script_dir, 'train_test_split.json'), # Same directory as script
        split='train',
        n_input_samples=6,
        sampler='random',
        return_masks=True,
        return_paths=True,
        random_seed=42
    )
    
    print(f"\nDataset size: {len(dataset_json)} samples")
    print("\n" + "=" * 80)
    print("Displaying 10 sample ROIs:")
    print("=" * 80)
    
    n_samples_to_show = min(10, len(dataset_json))
    
    for i in range(n_samples_to_show):
        print(f"\n[Sample {i+1}/{n_samples_to_show}]")
        print("-" * 80)
        
        try:
            sample = dataset_json[i]
            
            # ROI information
            print(f"ROI: {sample['input']['roi_name']}")
            
            # Shape information
            print(f"Input shape: {sample['input']['S2'].shape} (T={sample['input']['S2'].shape[0]}, C={sample['input']['S2'].shape[1]}, H={sample['input']['S2'].shape[2]}, W={sample['input']['S2'].shape[3]})")
            print(f"Target shape: {sample['target']['S2'].shape} (C={sample['target']['S2'].shape[0]}, H={sample['target']['S2'].shape[1]}, W={sample['target']['S2'].shape[2]})")
            
            # Input paths
            if 'paths' in sample['input']:
                print(f"\nInput images ({len(sample['input']['paths'])} files):")
                for idx, path in enumerate(sample['input']['paths'], 1):
                    print(f"  {idx}. {path}")
            
            # Target path
            if 'path' in sample['target']:
                print(f"\nTarget image:")
                print(f"  {sample['target']['path']}")
        
        except Exception as e:
            print(f"Error loading sample: {e}")
    
    print("\n" + "=" * 80)
    