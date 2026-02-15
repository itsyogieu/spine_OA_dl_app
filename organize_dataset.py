"""
Dataset Organizer and Splitter for Spine X-ray Images

This script helps organize spine X-ray images into train/val/test splits.

Usage:
    python organize_dataset.py --input raw_data/ --output dataset/ --split 0.7 0.15 0.15
"""

import os
import shutil
import argparse
from pathlib import Path
import random
from collections import defaultdict

def organize_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Organize images into train/val/test splits
    
    Args:
        input_dir: Path to raw images organized by grade
        output_dir: Path to output dataset folder
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Expected grade folders
    grades = ['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4']
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        for grade in grades:
            (output_path / split / grade).mkdir(parents=True, exist_ok=True)
    
    # Process each grade
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    for grade in grades:
        grade_path = input_path / grade
        
        if not grade_path.exists():
            print(f"⚠️  Warning: {grade} folder not found in input directory")
            continue
        
        # Get all images in this grade
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(grade_path.glob(ext)))
        
        if not image_files:
            print(f"⚠️  Warning: No images found in {grade}")
            continue
        
        # Shuffle images
        random.shuffle(image_files)
        
        # Calculate split indices
        n_images = len(image_files)
        train_idx = int(n_images * train_ratio)
        val_idx = train_idx + int(n_images * val_ratio)
        
        # Split images
        train_images = image_files[:train_idx]
        val_images = image_files[train_idx:val_idx]
        test_images = image_files[val_idx:]
        
        # Copy images to appropriate folders
        for img in train_images:
            shutil.copy2(img, output_path / 'train' / grade / img.name)
            stats[grade]['train'] += 1
        
        for img in val_images:
            shutil.copy2(img, output_path / 'val' / grade / img.name)
            stats[grade]['val'] += 1
        
        for img in test_images:
            shutil.copy2(img, output_path / 'test' / grade / img.name)
            stats[grade]['test'] += 1
        
        print(f"✅ {grade}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET ORGANIZATION COMPLETE")
    print("="*60)
    print(f"{'Grade':<15} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-"*60)
    
    for grade in grades:
        train_count = stats[grade]['train']
        val_count = stats[grade]['val']
        test_count = stats[grade]['test']
        total = train_count + val_count + test_count
        print(f"{grade:<15} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")
    
    print("="*60)
    total_train = sum(stats[g]['train'] for g in grades)
    total_val = sum(stats[g]['val'] for g in grades)
    total_test = sum(stats[g]['test'] for g in grades)
    total_all = total_train + total_val + total_test
    print(f"{'TOTAL':<15} {total_train:<10} {total_val:<10} {total_test:<10} {total_all:<10}")
    print("="*60)
    
    return stats

def verify_dataset(dataset_dir):
    """Verify the dataset structure and print statistics"""
    
    dataset_path = Path(dataset_dir)
    
    print("\nVerifying dataset structure...")
    print("="*60)
    
    for split in ['train', 'val', 'test']:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"❌ {split} folder not found!")
            continue
        
        print(f"\n{split.upper()}:")
        for grade in ['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4']:
            grade_path = split_path / grade
            if grade_path.exists():
                n_images = len(list(grade_path.glob('*.*')))
                print(f"  {grade}: {n_images} images")
            else:
                print(f"  {grade}: NOT FOUND")
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize spine X-ray dataset")
    parser.add_argument('--input', type=str, required=True, 
                       help='Input directory with grade folders')
    parser.add_argument('--output', type=str, default='dataset',
                       help='Output directory for organized dataset')
    parser.add_argument('--train', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing dataset, do not organize')
    
    args = parser.parse_args()
    
    # Verify ratios sum to 1
    if abs(args.train + args.val + args.test - 1.0) > 0.01:
        print("❌ Error: train + val + test ratios must sum to 1.0")
        exit(1)
    
    if args.verify_only:
        verify_dataset(args.output)
    else:
        print(f"Organizing dataset from {args.input} to {args.output}")
        print(f"Split ratios: train={args.train}, val={args.val}, test={args.test}")
        print(f"Random seed: {args.seed}")
        print("="*60)
        
        stats = organize_dataset(
            args.input, 
            args.output,
            args.train,
            args.val,
            args.test,
            args.seed
        )
        
        print("\n✅ Dataset organization complete!")
        print(f"📁 Output directory: {args.output}")
