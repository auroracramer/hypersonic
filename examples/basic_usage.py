#!/usr/bin/env python3
"""Basic usage example for HyperFuture."""

import argparse
import torch
import torch.nn as nn
from typing import Dict, Any

# Import HyperFuture components
import models
import datasets
from trainer import Trainer
from utils.utils import AverageMeter


def create_sample_args() -> argparse.Namespace:
    """Create sample arguments for demonstration."""
    args = argparse.Namespace()
    
    # Model parameters
    args.seq_len = 8
    args.img_dim = 224
    args.network_feature = 'resnet18'
    args.not_track_running_stats = False
    args.feature_dim = 512
    args.hyperbolic = False
    args.hyperbolic_version = 1
    args.num_seq = 4
    args.pred_step = 2
    args.distance = 'regular'
    args.cross_gpu_score = False
    args.use_labels = False
    args.hierarchical_labels = False
    args.pred_future = False
    args.early_action = False
    args.early_action_self = False
    args.dataset = 'kinetics'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.batch_size = 4
    args.lr = 1e-3
    args.epochs = 1
    
    return args


def demonstrate_model_creation():
    """Demonstrate how to create and use a HyperFuture model."""
    print("🚀 HyperFuture Basic Usage Example")
    print("=" * 50)
    
    # Create sample arguments
    args = create_sample_args()
    
    print(f"Device: {args.device}")
    print(f"Model: {args.network_feature}")
    print(f"Hyperbolic mode: {args.hyperbolic}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Number of sequences: {args.num_seq}")
    print()
    
    # Create model
    print("📦 Creating model...")
    model = models.Model(args)
    model = model.to(args.device)
    print(f"Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Create dummy data
    print("🔄 Creating dummy data...")
    batch_size = args.batch_size
    channels = 3
    seq_len = args.seq_len
    height = args.img_dim
    width = args.img_dim
    
    # Input shape: [B, N, C, SL, H, W]
    dummy_input = torch.randn(
        batch_size, args.num_seq, channels, seq_len, height, width
    ).to(args.device)
    
    print(f"Input shape: {dummy_input.shape}")
    print()
    
    # Forward pass
    print("⚡ Running forward pass...")
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output type: {type(output)}")
    if isinstance(output, (list, tuple)):
        print(f"Number of outputs: {len(output)}")
        for i, out in enumerate(output):
            if hasattr(out, 'shape'):
                print(f"Output {i} shape: {out.shape}")
    print()
    
    # Demonstrate utility functions
    print("🛠️  Demonstrating utility functions...")
    meter = AverageMeter()
    
    # Simulate some training metrics
    for i in range(5):
        loss_val = torch.randn(1).item()
        meter.update(loss_val)
        print(f"Step {i+1}: Loss = {loss_val:.4f}, Average = {meter.avg:.4f}")
    
    print()
    print("✅ Basic usage demonstration completed!")


def demonstrate_dataset_info():
    """Demonstrate dataset information."""
    print("📊 Dataset Information")
    print("=" * 30)
    
    # Show available datasets
    print("Available datasets:")
    print("- Kinetics600: Large-scale video dataset")
    print("- FineGym: Fine-grained gym activity dataset") 
    print("- MovieNet: Large-scale movie dataset")
    print("- Hollywood2: Movie action dataset")
    print()
    
    # Show dataset hierarchies
    print("Dataset hierarchies:")
    for dataset, info in datasets.sizes_hierarchy.items():
        print(f"- {dataset}: {info[0]} classes, hierarchy: {info[1]}")
    print()


def main():
    """Main function."""
    print("🎯 HyperFuture Package Demo")
    print("=" * 60)
    print()
    
    try:
        demonstrate_dataset_info()
        demonstrate_model_creation()
        
        print("🎉 All demonstrations completed successfully!")
        print("\nNext steps:")
        print("1. Download datasets from the links in README.md")
        print("2. Run: python main.py --help for full usage")
        print("3. Check examples in scripts/ directory")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please check your installation and try again.")


if __name__ == "__main__":
    main()