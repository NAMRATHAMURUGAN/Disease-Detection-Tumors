#!/usr/bin/env python3
"""
Check dataset structure and provide helpful information
"""
import os
import sys

def check_dataset_structure():
    """Check if dataset directories exist and show structure"""
    
    print("🔍 Checking Dataset Structure")
    print("=" * 50)
    
    # Check classification dataset
    classification_dir = "data/classification_dataset"
    if os.path.exists(classification_dir):
        print(f"✅ Classification dataset found: {classification_dir}")
        
        # List subdirectories
        subdirs = [d for d in os.listdir(classification_dir) 
                  if os.path.isdir(os.path.join(classification_dir, d))]
        
        print(f"   Subdirectories: {subdirs}")
        
        # Check for training/testing
        expected_dirs = ["Training", "Testing", "train", "test"]
        found_dirs = [d for d in expected_dirs if d in subdirs]
        
        if found_dirs:
            print(f"   ✅ Found data splits: {found_dirs}")
            
            for split_dir in found_dirs:
                split_path = os.path.join(classification_dir, split_dir)
                class_dirs = [d for d in os.listdir(split_path) 
                            if os.path.isdir(os.path.join(split_path, d))]
                print(f"   📁 {split_dir}/ contains: {class_dirs}")
                
                # Count images in each class
                for class_dir in class_dirs:
                    class_path = os.path.join(split_path, class_dir)
                    image_files = [f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
                    print(f"      📸 {class_dir}: {len(image_files)} images")
        else:
            print("   ❌ No training/testing directories found!")
            print("   Expected: 'Training' and 'Testing' (or 'train' and 'test')")
            
    else:
        print(f"❌ Classification dataset not found: {classification_dir}")
        print("   Expected structure:")
        print("   data/classification_dataset/")
        print("   ├── Training/")
        print("   │   ├── glioma/")
        print("   │   ├── meningioma/")
        print("   │   ├── pituitary/")
        print("   │   └── notumor/")
        print("   └── Testing/")
        print("       ├── glioma/")
        print("       ├── meningioma/")
        print("       ├── pituitary/")
        print("       └── notumor/")
    
    print()
    
    # Check segmentation dataset
    brats_dir = "data/brats_segmentation"
    if os.path.exists(brats_dir):
        print(f"✅ BraTS dataset found: {brats_dir}")
        
        subdirs = [d for d in os.listdir(brats_dir) 
                  if os.path.isdir(os.path.join(brats_dir, d))]
        print(f"   Subdirectories: {subdirs[:5]}{'...' if len(subdirs) > 5 else ''}")
        
        if subdirs:
            # Check first patient folder
            first_patient = os.path.join(brats_dir, subdirs[0])
            patient_files = os.listdir(first_patient)
            print(f"   📁 Sample patient files: {patient_files[:5]}")
            
    else:
        print(f"❌ BraTS dataset not found: {brats_dir}")
        print("   Expected structure:")
        print("   data/brats_segmentation/")
        print("   ├── BraTS2020_TrainingData/")
        print("   │   └── BraTS20_Training_001/")
        print("   │       ├── BraTS20_Training_001_t1.nii.gz")
        print("   │       ├── BraTS20_Training_001_t1ce.nii.gz")
        print("   │       ├── BraTS20_Training_001_t2.nii.gz")
        print("   │       ├── BraTS20_Training_001_flair.nii.gz")
        print("   │       └── BraTS20_Training_001_seg.nii.gz")
        print("   └── BraTS2020_ValidationData/")
    
    print()
    
    # Check survival dataset
    survival_dir = "data/survival_dataset"
    if os.path.exists(survival_dir):
        print(f"✅ Survival dataset found: {survival_dir}")
        
        files = os.listdir(survival_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        print(f"   CSV files: {csv_files}")
        
    else:
        print(f"❌ Survival dataset not found: {survival_dir}")
        print("   Expected structure:")
        print("   data/survival_dataset/")
        print("   └── clinical_data.csv")
    
    print()
    print("📋 Summary:")
    print("   ✅ Dataset structure check completed")
    print("   💡 If datasets are missing, please download and organize them as shown above")
    print("   🚀 Once datasets are ready, you can start training with:")
    print("      python -m src.train_occurrence")
    print("      python -m src.train_classifier")
    print("      python -m src.train_unet")

if __name__ == "__main__":
    check_dataset_structure()
