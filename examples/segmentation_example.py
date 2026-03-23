"""
Example script demonstrating tumor segmentation usage
"""
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.segmentation_dataset import create_segmentation_datasets
from src.models.segmentation.unet import create_unet_model
from src.inference.segmentation_inference import create_segmentation_inference


def main():
    """Example usage of tumor segmentation"""
    
    print("=== Tumor Segmentation Example ===\n")
    
    # 1. Create datasets
    print("1. Creating datasets from BraTS data...")
    try:
        train_dataset, val_dataset = create_segmentation_datasets(
            root_dir="data/brats_segmentation",
            val_split=0.2,
            image_size=256,
            modalities=["t1", "t1ce", "t2", "flair"],
            augment=True,
            seed=42
        )
        
        print(f"   Training dataset: {len(train_dataset)} slices")
        print(f"   Validation dataset: {len(val_dataset)} slices")
        
        # Show class distribution
        class_dist = train_dataset.get_class_distribution()
        class_names = ['background', 'necrotic', 'edema', 'enhancing']
        print("   Class distribution:")
        for class_idx, count in class_dist.items():
            print(f"     {class_names[class_idx]}: {count:,} voxels")
        
    except Exception as e:
        print(f"   Error creating datasets: {str(e)}")
        print("   Make sure BraTS dataset is available in data/brats_segmentation/")
        return
    
    # 2. Create model
    print("\n2. Creating 2D U-Net model...")
    model = create_unet_model(
        in_channels=4,
        num_classes=4,
        base_filters=64,
        bilinear=True,
        dropout=0.1,
        attention=False
    )
    
    # Print model info
    from src.utils.helpers import print_model_info
    print_model_info(model, "2D U-Net")
    
    # 3. Test with sample data
    print("\n3. Testing model with sample data...")
    sample_image, sample_mask = train_dataset[0]
    print(f"   Sample image shape: {sample_image.shape}")
    print(f"   Sample mask shape: {sample_mask.shape}")
    print(f"   Unique mask values: {torch.unique(sample_mask).tolist()}")
    
    # Forward pass
    with torch.no_grad():
        model.eval()
        output = model(sample_image.unsqueeze(0))
        print(f"   Model output shape: {output.shape}")
        print(f"   Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # 4. Inference example (if model is trained)
    print("\n4. Inference example...")
    model_path = "checkpoints/unet/best_model.pth"
    
    if os.path.exists(model_path):
        print("   Loading trained model for inference...")
        inference = create_segmentation_inference(model_path)
        
        # Find a sample patient folder
        import glob
        patient_folders = glob.glob("data/brats_segmentation/**/BraTS20_*", recursive=True)
        
        if patient_folders:
            sample_patient = patient_folders[0]
            print(f"   Processing sample patient: {os.path.basename(sample_patient)}")
            
            try:
                # Predict single slice
                modality_volumes = {}
                for modality in ["t1", "t1ce", "t2", "flair"]:
                    modality_file = glob.glob(os.path.join(sample_patient, f"*{modality}*.nii*"))[0]
                    import nibabel as nib
                    img = nib.load(modality_file)
                    modality_volumes[modality] = img.get_fdata()
                
                # Extract middle axial slice
                slice_idx = modality_volumes["t1"].shape[2] // 2
                modality_slices = []
                for modality in ["t1", "t1ce", "t2", "flair"]:
                    slice_data = modality_volumes[modality][:, :, slice_idx]
                    modality_slices.append(slice_data)
                
                # Predict slice
                result = inference.predict_slice(modality_slices, return_probabilities=True)
                print(f"   Slice {slice_idx} prediction shape: {result['prediction'].shape}")
                print(f"   Unique prediction values: {torch.unique(torch.from_numpy(result['prediction'])).tolist()}")
                
                # Calculate metrics if ground truth available
                try:
                    seg_file = glob.glob(os.path.join(sample_patient, "*seg*.nii*"))[0]
                    seg_img = nib.load(seg_file)
                    seg_data = seg_img.get_fdata()
                    seg_data[seg_data == 4] = 3  # Remap label 4 to 3
                    
                    ground_truth = seg_data[:, :, slice_idx]
                    from src.evaluation.segmentation_metrics import SegmentationMetrics
                    metrics = SegmentationMetrics()
                    
                    pred_tensor = torch.from_numpy(result['prediction']).unsqueeze(0).unsqueeze(0)
                    gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0).unsqueeze(0)
                    
                    dice = metrics.dice_coefficient(pred_tensor, gt_tensor)
                    print(f"   Dice coefficient: {dice.mean().item():.4f}")
                    
                except Exception as e:
                    print(f"   Could not calculate metrics: {str(e)}")
                
            except Exception as e:
                print(f"   Error during inference: {str(e)}")
        else:
            print("   No patient folders found for inference example")
    else:
        print(f"   No trained model found at {model_path}")
        print("   Train the model first using: python scripts/train_segmentation.py")
    
    # 5. Training setup example
    print("\n5. Training setup example...")
    print("   To train the model, run:")
    print("   python scripts/train_segmentation.py \\")
    print("       --data_path data/brats_segmentation \\")
    print("       --epochs 100 \\")
    print("       --batch_size 8 \\")
    print("       --learning_rate 1e-4")
    
    print("\n=== Example completed ===")


if __name__ == '__main__':
    import torch
    main()
