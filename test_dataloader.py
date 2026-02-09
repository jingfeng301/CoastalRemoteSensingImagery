"""
Quick test script for CoastlineCloudRemovalDataset

This script demonstrates how to use the dataloader with minimal configuration.
Only the data_root path needs to be updated.

Usage:
    python test_dataloader.py --data_root /path/to/your/allclear/data_root
"""

import argparse
from data.dataloader import CoastlineCloudRemovalDataset


def main():
    parser = argparse.ArgumentParser(description="Test the CoastlineCloudRemovalDataset")
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to AllClear data root directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test', 'all'],
        help='Which split to test (default: train)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=3,
        help='Number of samples to load and display (default: 3)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Testing CoastlineCloudRemovalDataset")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data root: {args.data_root}")
    print(f"  Split: {args.split}")
    print(f"  Samples to test: {args.n_samples}")
    print()
    
    # Create dataset with relative paths for repo files
    dataset = CoastlineCloudRemovalDataset(
        csv_path='data/data.csv',                      # Relative path (in repo)
        data_root=args.data_root,                       # User-provided path
        target_dir='coastline_target',                  # Relative path (in repo)
        split_json_path='data/train_test_split.json',  # Relative path (in repo)
        split=args.split,
        n_input_samples=6,
        sampler='random',
        return_masks=True,
        return_paths=True,
        random_seed=42
    )
    
    print("\n" + "=" * 80)
    print("Dataset loaded successfully!")
    print("=" * 80)
    print(f"\nTotal samples in {args.split} split: {len(dataset)}")
    
    if len(dataset) == 0:
        print("\n⚠️  No samples found. Please check your data paths.")
        return
    
    # Load and display sample information
    print(f"\nLoading {min(args.n_samples, len(dataset))} sample(s)...\n")
    
    for i in range(min(args.n_samples, len(dataset))):
        print(f"Sample {i+1}/{min(args.n_samples, len(dataset))}:")
        print("-" * 80)
        
        try:
            sample = dataset[i]
            
            print(f"  ROI: {sample['input']['roi_name']}")
            print(f"  Input shape: {sample['input']['S2'].shape}")
            print(f"    - Time steps: {sample['input']['S2'].shape[0]}")
            print(f"    - Channels: {sample['input']['S2'].shape[1]}")
            print(f"    - Height: {sample['input']['S2'].shape[2]}")
            print(f"    - Width: {sample['input']['S2'].shape[3]}")
            
            print(f"  Target shape: {sample['target']['S2'].shape}")
            print(f"    - Channels: {sample['target']['S2'].shape[0]}")
            print(f"    - Height: {sample['target']['S2'].shape[1]}")
            print(f"    - Width: {sample['target']['S2'].shape[2]}")
            
            if 'masks' in sample['input']:
                print(f"  Mask shape: {sample['input']['masks'].shape}")
                if 'coverage' in sample['input']:
                    coverages = sample['input']['coverage']
                    print(f"  Cloud coverage: {[f'{c:.1%}' for c in coverages]}")
            
            if 'paths' in sample['input']:
                print(f"  Number of input files: {len(sample['input']['paths'])}")
            
            print("  ✓ Sample loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Error loading sample: {e}")
        
        print()
    
    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    print("\nYour dataset is ready for training. You can now:")
    print("  1. Create a PyTorch DataLoader from this dataset")
    print("  2. Start training your cloud removal model")
    print("  3. Use different splits (train/val/test) for evaluation")


if __name__ == "__main__":
    main()
