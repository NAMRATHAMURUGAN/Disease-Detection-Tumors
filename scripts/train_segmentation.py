"""
Training script for tumor segmentation
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.configs.segmentation_config import SegmentationConfig
from src.data.segmentation_dataset import create_segmentation_datasets
from src.models.segmentation.unet import create_unet_model
from src.training.segmentation_trainer import SegmentationTrainer
from src.utils.helpers import set_seed, save_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train tumor segmentation model')
    
    parser.add_argument('--data_path', type=str, default='data/brats_segmentation',
                       help='Path to BraTS dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (use smaller for segmentation)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--base_filters', type=int, default=64,
                       help='Base number of filters in U-Net')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--attention', action='store_true',
                       help='Use attention U-Net')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                       help='Weight for Dice loss')
    parser.add_argument('--ce_weight', type=float, default=1.0,
                       help='Weight for Cross-Entropy loss')
    parser.add_argument('--modalities', type=str, nargs='+',
                       default=['t1', 't1ce', 't2', 'flair'],
                       help='MRI modalities to use')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create configuration
    config = SegmentationConfig()
    config.dataset_path = args.data_path
    config.checkpoint_dir = args.checkpoint_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.image_size = args.image_size
    config.base_filters = args.base_filters
    config.num_workers = args.num_workers
    config.seed = args.seed
    config.device = device
    config.resume = args.resume
    config.modalities = args.modalities
    config.dice_weight = args.dice_weight
    config.ce_weight = args.ce_weight
    
    # Set random seed
    set_seed(config.seed)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_segmentation_datasets(
        root_dir=config.dataset_path,
        val_split=0.2,
        image_size=config.image_size,
        modalities=config.modalities,
        augment=True,
        seed=config.seed
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Modalities: {config.modalities}")
    
    # Print class distribution
    class_dist = train_dataset.get_class_distribution()
    print("Class distribution (training):")
    for class_idx, count in class_dist.items():
        class_names = ['background', 'necrotic', 'edema', 'enhancing']
        print(f"  {class_names[class_idx]}: {count:,} voxels")
    
    # Create model
    print("Creating model...")
    model = create_unet_model(
        in_channels=len(config.modalities),
        num_classes=config.num_classes,
        base_filters=config.base_filters,
        bilinear=True,
        dropout=0.1,
        attention=args.attention
    )
    
    # Print model info
    from src.utils.helpers import print_model_info
    model_name = "Attention U-Net" if args.attention else "U-Net"
    print_model_info(model, model_name)
    
    # Create trainer
    print("Creating trainer...")
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Save configuration
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    save_config(config, os.path.join(config.checkpoint_dir, 'segmentation_config.json'))
    
    # Start training
    print("Starting training...")
    results = trainer.train()
    
    # Print final results
    print("\nTraining completed!")
    print(f"Best validation Dice: {results['best_val_dice']:.4f}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    
    # Save final results
    import json
    results_path = os.path.join(config.checkpoint_dir, 'segmentation_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # Print patient information
    print("\nDataset Information:")
    patient_info = train_dataset.get_patient_info()
    print(f"Total patients: {len(patient_info)}")
    
    total_slices = sum(p['total_slices'] for p in patient_info)
    tumor_slices = sum(p['tumor_slices'] for p in patient_info)
    
    print(f"Total slices: {total_slices:,}")
    print(f"Tumor slices: {tumor_slices:,}")
    print(f"Tumor slice ratio: {tumor_slices/total_slices:.2%}")


if __name__ == '__main__':
    main()
