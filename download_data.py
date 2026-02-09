"""
Download coastline_target data from MEGA.nz

This script downloads the coastline_target folder containing 10,223 target images
from MEGA.nz storage.

Usage:
    python download_data.py [--output_dir OUTPUT_DIR]

Arguments:
    --output_dir: Directory where the data will be downloaded (default: current directory)

Requirements:
    - megatools (install via: sudo apt install megatools)
    OR
    - Download manually from the MEGA.nz link
"""

import os
import sys
import argparse
import zipfile
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm

# MEGA.nz file link
MEGA_URL = "https://mega.nz/file/VxsDSLKB#Fnb7HVV3Gq57ontClQq4BYvR3U9fa5QPoPXXCVdsDDI"


def check_megatools_installed():
    """
    Check if megatools is installed and available
    
    Returns:
        bool: True if megadl is available, False otherwise
    """
    try:
        result = subprocess.run(['megadl', '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def print_installation_instructions():
    """
    Print instructions for installing megatools
    """
    print("\n" + "=" * 80)
    print("MEGATOOLS NOT FOUND")
    print("=" * 80)
    print("\nTo download from MEGA.nz automatically, please install megatools:\n")
    print("Ubuntu/Debian:")
    print("  sudo apt update")
    print("  sudo apt install megatools\n")
    print("Fedora/RHEL:")
    print("  sudo dnf install megatools\n")
    print("Arch Linux:")
    print("  sudo pacman -S megatools\n")
    print("macOS (using Homebrew):")
    print("  brew install megatools\n")
    print("After installation, run this script again.")
    print("\n" + "-" * 80)
    print("ALTERNATIVE: Manual Download")
    print("-" * 80)
    print(f"\n1. Open this link in your browser:\n   {MEGA_URL}")
    print(f"\n2. Download the file to: {os.path.abspath('.')}")
    print("\n3. Extract the archive if needed")
    print("=" * 80 + "\n")


def download_from_mega(mega_url, output_dir):
    """
    Download file from MEGA.nz using megadl command-line tool
    
    Args:
        mega_url (str): MEGA.nz public file link
        output_dir (str): Directory to save the downloaded file
    """
    print(f"Downloading coastline_target data from MEGA.nz...")
    print(f"URL: {mega_url}")
    print(f"Output directory: {output_dir}")
    print("-" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if megatools is installed
    if not check_megatools_installed():
        print_installation_instructions()
        sys.exit(1)
    
    try:
        # Download the file using megadl
        print("\nStarting download... This may take several minutes depending on file size.")
        print(f"Running: megadl --path {output_dir} {mega_url}")
        print()
        
        # Run megadl with real-time output (megadl shows its own progress)
        result = subprocess.run(
            ['megadl', '--path', output_dir, mega_url],
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print("\n✓ Download completed successfully!")
            # Find the downloaded file
            files = [f for f in os.listdir(output_dir) 
                    if os.path.isfile(os.path.join(output_dir, f))]
            
            if files:
                # Get the most recently modified file
                files_with_time = [(f, os.path.getmtime(os.path.join(output_dir, f))) 
                                  for f in files]
                latest_file = max(files_with_time, key=lambda x: x[1])[0]
                file_path = os.path.join(output_dir, latest_file)
                print(f"Downloaded file: {file_path}")
                return file_path
            else:
                print("✗ No file found after download")
                sys.exit(1)
        else:
            print(f"\n✗ Error downloading file (exit code: {result.returncode})")
            print("\n" + "=" * 80)
            print("Please try downloading manually from:")
            print(f"  {mega_url}")
            print("=" * 80)
            sys.exit(1)
    
    except subprocess.TimeoutExpired:
        print("✗ Download timed out (exceeded 1 hour)")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error downloading file: {e}")
        print("\n" + "=" * 80)
        print("Please try downloading manually from:")
        print(f"  {mega_url}")
        print("=" * 80)
        sys.exit(1)


def extract_archive(file_path, output_dir):
    """
    Extract downloaded archive if it's a zip, tar.gz, or rar file
    
    Args:
        file_path (str): Path to the downloaded file
        output_dir (str): Directory to extract to
    """
    if file_path.endswith('.zip'):
        print(f"\nExtracting ZIP archive: {file_path}")
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # Get list of files in the archive
                file_list = zip_ref.namelist()
                
                # Extract with progress bar
                with tqdm(total=len(file_list), desc="Extracting", unit="file") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, output_dir)
                        pbar.update(1)
            
            print(f"✓ Extraction completed to: {output_dir}")
            
            # Automatically remove the zip file after extraction
            print(f"Removing archive: {file_path}")
            os.remove(file_path)
            print(f"✓ Archive removed successfully")
        
        except Exception as e:
            print(f"✗ Error extracting archive: {e}")
            sys.exit(1)
    
    elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
        print(f"\nExtracting TAR archive: {file_path}")
        import tarfile
        try:
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                members = tar_ref.getmembers()
                
                # Extract with progress bar
                with tqdm(total=len(members), desc="Extracting", unit="file") as pbar:
                    for member in members:
                        tar_ref.extract(member, output_dir)
                        pbar.update(1)
            
            print(f"✓ Extraction completed to: {output_dir}")
            
            # Automatically remove the tar file after extraction
            print(f"Removing archive: {file_path}")
            os.remove(file_path)
            print(f"✓ Archive removed successfully")
        
        except Exception as e:
            print(f"✗ Error extracting archive: {e}")
            sys.exit(1)
    
    elif file_path.endswith('.rar'):
        print(f"\nExtracting RAR archive: {file_path}")
        
        # Check if unrar is installed
        try:
            subprocess.run(['unrar'], capture_output=True, timeout=2)
        except FileNotFoundError:
            print("\n⚠ 'unrar' is not installed.")
            print("Installing unrar...")
            try:
                subprocess.run(['sudo', 'apt', 'install', '-y', 'unrar'], check=True)
                print("✓ unrar installed successfully!")
            except subprocess.CalledProcessError:
                print("✗ Failed to install unrar.")
                print("Please install manually: sudo apt install unrar")
                sys.exit(1)
        
        try:
            # Extract with unrar
            result = subprocess.run(
                ['unrar', 'x', '-y', file_path, output_dir],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                print(f"✓ Extraction completed to: {output_dir}")
                
                # Automatically remove the rar file after extraction
                print(f"Removing archive: {file_path}")
                os.remove(file_path)
                print(f"✓ Archive removed successfully")
            else:
                print(f"✗ Error extracting RAR archive")
                print(f"Error: {result.stderr}")
                sys.exit(1)
        
        except subprocess.TimeoutExpired:
            print("✗ Extraction timed out")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error extracting archive: {e}")
            sys.exit(1)
    
    else:
        print(f"\n✓ File downloaded successfully: {file_path}")
        print("Note: File is not an archive, no extraction needed.")


def verify_download(output_dir):
    """
    Verify that the coastline_target folder exists and contains data
    
    Args:
        output_dir (str): Directory where data was downloaded
    """
    coastline_target_dir = os.path.join(output_dir, 'coastline_target')
    
    if os.path.exists(coastline_target_dir) and os.path.isdir(coastline_target_dir):
        # Count the number of files in the directory with progress
        print("\nVerifying downloaded files...")
        all_items = os.listdir(coastline_target_dir)
        file_count = 0
        
        for item in tqdm(all_items, desc="Checking files", unit="item"):
            if os.path.isfile(os.path.join(coastline_target_dir, item)):
                file_count += 1
        
        print("\n" + "=" * 80)
        print("✓ SUCCESS: coastline_target folder found!")
        print(f"  Location: {coastline_target_dir}")
        print(f"  Contains: {file_count} files")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print("⚠ WARNING: coastline_target folder not found in expected location.")
        print(f"  Expected: {coastline_target_dir}")
        print("  Please check the extracted contents manually.")
        print("=" * 80)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download coastline_target data from MEGA.nz"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory where the data will be downloaded (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path
    output_dir = os.path.abspath(args.output_dir)
    
    print("=" * 80)
    print("Coastal Remote Sensing Imagery Dataset - Data Downloader")
    print("=" * 80)
    print()
    
    # Download from MEGA
    downloaded_file = download_from_mega(MEGA_URL, output_dir)
    
    # Extract if it's an archive
    extract_archive(downloaded_file, output_dir)
    
    # Verify the download
    verify_download(output_dir)
    
    print("\nDownload process completed!")
    print(f"Files are located in: {output_dir}")


if __name__ == "__main__":
    main()
