"""
Training script for tumor classification
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.configs.classification_config import ClassificationConfig
from src.data.classification_dataset import TumorClassificationDataset, get_classification_transforms
from src.models.classification.resnet_classifier import create_resnet_classifier
from src.training.classification_trainer import ClassificationTrainer
from src.utils.helpers import set_seed, save_config


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train tumor classification model')
    
    parser.add_argument('--data_path', type=str, default='data/classification_dataset',
                       help='Path to classification dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--model_name', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from checkpoint')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone for feature extraction')
    
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
    config = ClassificationConfig()
    config.dataset_path = args.data_path
    config.checkpoint_dir = args.checkpoint_dir
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.model_name = args.model_name
    config.image_size = args.image_size
    config.num_workers = args.num_workers
    config.seed = args.seed
    config.device = device
    config.resume = args.resume
    
    # Set random seed
    set_seed(config.seed)
    
    # Create transforms
    transforms = get_classification_transforms(config.image_size)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TumorClassificationDataset(
        root_dir=config.dataset_path,
        transform=transforms['train'],
        split='train',
        val_split=0.2,
        seed=config.seed
    )
    
    val_dataset = TumorClassificationDataset(
        root_dir=config.dataset_path,
        transform=transforms['val'],
        split='val',
        val_split=0.2,
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
    print(f"Classes: {train_dataset.classes}")
    
    # Create model
    print("Creating model...")
    model = create_resnet_classifier(
        num_classes=config.num_classes,
        model_name=config.model_name,
        pretrained=True,
        freeze_backbone=args.freeze_backbone
    )
    
    # Print model info
    from src.utils.helpers import print_model_info
    print_model_info(model, f"ResNet-{config.model_name.split('resnet')[1]}")
    
    # Create trainer
    print("Creating trainer...")
    trainer = ClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Save configuration
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    save_config(config, os.path.join(config.checkpoint_dir, 'config.json'))
    
    # Start training
    print("Starting training...")
    results = trainer.train()
    
    # Print final results
    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"Training time: {results['training_time']:.2f} seconds")
    
    # Save final results
    import json
    results_path = os.path.join(config.checkpoint_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
