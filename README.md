# Coastal Remote Sensing Imagery Dataset

A large-scale dataset for cloud removal in coastal Sentinel-2 imagery, containing 10,223 regions of interest (ROIs) derived from the AllClear dataset with specialized coastal filtering.

## 📋 Overview

This dataset is designed for training and evaluating cloud removal models on coastal remote sensing imagery. Built upon the [AllClear dataset](https://github.com/Zhou-Hangyu/allclear), it applies custom filtering criteria to focus specifically on coastal regions.

### Dataset Construction Pipeline

1. **Base Dataset**: Started with AllClear's Sentinel-2 global dataset
2. **ROI Extraction**: Identified all ROIs from AllClear that have at least one valid target image and exist in the downloaded data root
3. **Coastal Filtering**: Applied custom geographic filtering to select only ROIs within 300km of global coastlines using Natural Earth shapefiles
4. **Quality Validation**: Verified data availability and integrity for each ROI
5. **Target Selection**: Applied preprocessing to ensure exactly one target per ROI:
   - Single-date targets (450 ROIs): Best quality single acquisition
   - Two-date targets (455 ROIs): Best path selected from two candidates  
   - Composite targets (9,318 ROIs): Optimal composites from 3+ acquisitions
6. **Final Dataset**: **10,223 coastal ROIs** with validated targets and temporal input sequences

### Dataset Features

- **10,223 Coastal ROIs** - Each with exactly one target image (cloud-free or optimal composite)
- **Multiple temporal input images per ROI** - Varying cloud coverage for temporal learning
- **Cloud/shadow probability masks** - Values 0-100% representing cloud confidence for all input images
- **Global coastal coverage** - Focused on coastal regions within 300km of coastlines
- **Quality-controlled** - All images validated for readability and data integrity (NaN < 10%)

### Geographic Distribution

The map below shows the global distribution of coastal ROIs (green) within 300km of coastlines. Red markers indicate ROIs from AllClear that were excluded for being beyond the 300km coastal threshold.

![Coastline ROIs Distribution](/home/wangyu/CoastalRemoteSensingImagery/CoastlineFilter.png)

*Figure: Global distribution of 10,223 coastal ROIs (green) and excluded inland ROIs (red) from the AllClear dataset. Blue lines represent global coastlines from Natural Earth shapefiles.*

## Setup & Installation

### Step 1: Environment Setup

Create a new Python environment and install dependencies:

```bash
# Create virtual environment (recommended)
conda create -n coastline_dataset python=3.9
conda activate coastline_dataset

# Or using venv
python -m venv coastline_env
source coastline_env/bin/activate  # On Windows: coastline_env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Required packages:**
- `torch` (PyTorch for deep learning)
- `rasterio` (GeoTIFF file reading)
- `pandas` (CSV data processing)
- `numpy` (Numerical operations)
- `tqdm` (Progress bars)
- `natsort` (Natural sorting)

### Step 2: Download Input Images (AllClear Dataset)

Download the complete AllClear dataset which contains the input images:

```bash
# Clone the AllClear repository
git clone https://github.com/Zhou-Hangyu/allclear.git
cd allclear

# Use their download script to get the full dataset
python download.py --output_dir /path/to/your/data_root
```

This will download all ROI directories with their temporal Sentinel-2 images and cloud probability masks (0-100%). This becomes your **data_root** for input images.

**Expected structure after download:**
```
/path/to/your/data_root/
├── roi954/
│   ├── 2022_1/
│   │   ├── s2_toa/      # Sentinel-2 TOA reflectance images
│   │   └── cld_shdw/    # Cloud probability masks (0-100%)
│   └── ...
├── roi263/
└── ...
```

### Step 3: Download Target Images

Download the consolidated target images (10,223 TIF files) using our automated download script:

#### Method 1: Automated Download (Recommended)

**Prerequisites:** Install megatools (command-line tool for MEGA.nz)

```bash
# Ubuntu/Debian
sudo apt update && sudo apt install megatools

# Fedora/RHEL
sudo dnf install megatools

# Arch Linux
sudo pacman -S megatools

# macOS (using Homebrew)
brew install megatools
```

**Download the data:**

```bash
# Run the download script
python download_data.py

# Or specify a custom output directory
python download_data.py --output_dir /path/to/your/data/directory
```

The script will:
- Download the coastline_target data from MEGA.nz (10,223 target images)
- Extract the archive automatically
- Verify the downloaded data

**MEGA.nz Link:** `https://mega.nz/file/VxsDSLKB#Fnb7HVV3Gq57ontClQq4BYvR3U9fa5QPoPXXCVdsDDI`

#### Method 2: Manual Download

Alternatively, you can manually download from the link above and extract:

```bash
# Extract to your project directory
# The archive contains the coastline_target/ folder with all target images
unzip coastline_targets.zip -d /path/to/CoastalRemoteSensingImagery/
```

**Expected structure after extraction:**
```
CoastalRemoteSensingImagery/
└── coastline_target/
    ├── roi954_s2_toa_2022_2_21_median.tif
    ├── roi263_optimal_composite.tif
    └── ... (10,223 TIF files total)
```

### Step 4: Verify Data Integrity

Before proceeding, verify that all data has been downloaded correctly:

```bash
python verify_dataloader.py --data_root /path/to/your/allclear/data_root
```

You can also customize the verification:
```bash
python verify_dataloader.py \
    --data_root /path/to/your/allclear/data_root \
    --target_dir coastline_target \
    --csv_path data/data.csv \
    --split_json_path data/train_test_split.json \
    --n_samples 20
```

This verification script will:
- ✓ Test dataset creation for all splits (train/val/test)
- ✓ Load sample data and check for errors
- ✓ Verify shape consistency across inputs/targets/masks
- ✓ Check value ranges and data quality (NaN, Inf, negatives)
- ✓ Validate mask alignment with inputs
- ✓ Check ROI consistency between inputs and targets
- ✓ Test all sampling modes (all, fixed, random)
- ✓ Verify geospatial coordinate consistency

**Expected output:**
```
🎉 ALL TESTS PASSED! Dataset is ready for training.
```

If any tests fail, the script will provide detailed error messages indicating what needs to be fixed.

### Step 5: Quick Test

Run a quick test to ensure the dataloader works:

```bash
python test_dataloader.py --data_root /path/to/your/allclear/data_root
```

This will load 3 samples and display their information to confirm everything is working correctly.

---

## 📊 Dataset Usage

### Basic Usage

```python
from data.dataloader import CoastlineCloudRemovalDataset

dataset = CoastlineCloudRemovalDataset(
    csv_path='data/data.csv',                        # Included in repo (no change needed)
    data_root='/path/to/your/allclear/data_root',   # UPDATE THIS: Path to AllClear data
    target_dir='coastline_target',                   # Included in repo (no change needed)
    split_json_path='data/train_test_split.json',   # Included in repo (no change needed)
    split='train'
)
```

**Only one path needs to be changed:**
- `data_root`: Path to where you downloaded the AllClear dataset in Step 2

All other files use relative paths and are included in the repository.

### Advanced Configuration

You can customize various dataset parameters:

```python
dataset = CoastlineCloudRemovalDataset(
    csv_path='data/data.csv',
    data_root='/path/to/your/allclear/data_root',
    target_dir='coastline_target',
    split_json_path='data/train_test_split.json',
    split='train',                    # 'train', 'val', 'test', or 'all'
    n_input_samples=6,                # Number of temporal inputs to use
    sampler='random',                 # 'random', 'fixed', or 'all'
    return_masks=True,                # Include cloud probability masks
    return_paths=True,                # Include file paths in output
    random_seed=42                    # Seed for reproducibility
)
```

For integration with PyTorch training loops, wrap the dataset in a DataLoader:

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset, 
    batch_size=8, 
    shuffle=True, 
    num_workers=4
)

for batch in dataloader:
    inputs = batch['input']['S2']      # [B, T, C, H, W]
    targets = batch['target']['S2']    # [B, C, H, W]
    masks = batch['input']['masks']    # [B, T, H, W]
    # Your training code here...
```

**Last Updated**: February 9, 2026
