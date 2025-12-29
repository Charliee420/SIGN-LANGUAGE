"""
Dataset Download Helper for ISL Detection
Downloads the Indian Sign Language dataset from Kaggle

IMPORTANT: You need a Kaggle account to download the dataset.
If you don't have one, follow these steps:
1. Go to https://www.kaggle.com/ and create an account
2. Go to your profile -> Settings -> API -> Create New Token
3. This will download a kaggle.json file
4. Place it in ~/.kaggle/ (Linux/Mac) or C:/Users/<username>/.kaggle/ (Windows)
"""
import os
import sys
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import DATA_DIR, BASE_DIR


def download_with_kagglehub():
    """
    Download dataset using kagglehub library
    """
    try:
        import kagglehub
        
        print("Downloading ISL dataset from Kaggle...")
        print("Dataset: indian-sign-language-islrtc-referred")
        print("-" * 50)
        
        # Download the dataset
        path = kagglehub.dataset_download("prathumarikeri/indian-sign-language-islrtc-referred")
        
        print(f"\n✓ Dataset downloaded to: {path}")
        
        # Move to our data directory
        print(f"\nOrganizing dataset to: {DATA_DIR}")
        
        # Check if the downloaded path has the expected structure
        if os.path.exists(path):
            # Copy contents to our data directory
            for item in os.listdir(path):
                src = os.path.join(path, item)
                dst = os.path.join(DATA_DIR, item)
                
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            
            print(f"✓ Dataset organized successfully!")
            return True
        
        return False
        
    except ImportError:
        print("kagglehub not installed. Installing...")
        os.system("pip install kagglehub")
        return download_with_kagglehub()
    
    except Exception as e:
        print(f"Error downloading with kagglehub: {e}")
        return False


def manual_download_instructions():
    """
    Show instructions for manual download
    """
    print("\n" + "=" * 60)
    print("   MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
If automatic download fails, please download manually:

1. Go to: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-islrtc-referred

2. Click "Download" button (you need to be logged in)

3. Extract the downloaded ZIP file

4. Copy the folders (0-9, A-Z) to:
   {data_dir}

The folder structure should look like:
   data/ISL_Dataset/
   ├── 0/
   │   ├── image1.jpg
   │   ├── image2.jpg
   │   └── ...
   ├── 1/
   ├── ...
   ├── 9/
   ├── A/
   ├── B/
   ├── ...
   └── Z/

Each folder should contain ~1000 images of that gesture.
""".format(data_dir=DATA_DIR))
    print("=" * 60)


def verify_dataset():
    """
    Verify the dataset structure
    """
    print("\nVerifying dataset structure...")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Data directory not found: {DATA_DIR}")
        return False
    
    expected_classes = [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]
    found_classes = []
    missing_classes = []
    
    for cls in expected_classes:
        cls_path = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_path) and os.path.isdir(cls_path):
            num_images = len([f for f in os.listdir(cls_path) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            if num_images > 0:
                found_classes.append((cls, num_images))
            else:
                missing_classes.append(cls)
        else:
            missing_classes.append(cls)
    
    print(f"\n✓ Found {len(found_classes)} classes:")
    for cls, count in found_classes[:5]:
        print(f"    {cls}: {count} images")
    if len(found_classes) > 5:
        print(f"    ... and {len(found_classes) - 5} more classes")
    
    total_images = sum(count for _, count in found_classes)
    print(f"\n✓ Total images: {total_images:,}")
    
    if missing_classes:
        print(f"\n⚠ Missing classes: {', '.join(missing_classes[:10])}")
        if len(missing_classes) > 10:
            print(f"    ... and {len(missing_classes) - 10} more")
    
    if len(found_classes) >= 20:
        print("\n✓ Dataset is ready for training!")
        return True
    else:
        print("\n❌ Dataset is incomplete. Please download the full dataset.")
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("   ISL Dataset Download Helper")
    print("=" * 60)
    
    # Check if dataset already exists
    if os.path.exists(DATA_DIR) and os.listdir(DATA_DIR):
        print(f"\nDataset directory exists: {DATA_DIR}")
        if verify_dataset():
            print("\nDataset already downloaded and verified!")
            choice = input("\nDo you want to re-download? (y/n): ").strip().lower()
            if choice != 'y':
                return
    
    # Attempt automatic download
    print("\nAttempting automatic download from Kaggle...")
    success = download_with_kagglehub()
    
    if success:
        verify_dataset()
        print("\n" + "=" * 60)
        print("   Next Steps:")
        print("   1. Train the model: python src/train.py")
        print("   2. Run detection:   python src/predict.py")
        print("=" * 60)
    else:
        manual_download_instructions()


if __name__ == "__main__":
    main()
