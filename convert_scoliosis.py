# Save as: convert_scoliosis_fixed.py

import os
import shutil
from pathlib import Path
import random

def convert_scoliosis_yolo_dataset():
    """Convert YOLO format scoliosis dataset to 5-grade classification"""
    
    print("="*60)
    print("CONVERTING SCOLIOSIS DATASET (YOLO Format)")
    print("="*60)
    
    raw_dir = Path('raw_data')
    output_dir = Path('dataset')
    
    # Check if raw_data exists
    if not raw_dir.exists():
        print(f"❌ raw_data folder not found!")
        return False
    
    # Create output structure
    print("\n📁 Creating grade folders...")
    for split in ['train', 'val', 'test']:
        for grade in ['grade_0', 'grade_1', 'grade_2', 'grade_3', 'grade_4']:
            (output_dir / split / grade).mkdir(parents=True, exist_ok=True)
    print("✅ Folders created!")
    
    # Find images - YOLO structure has images in "images" subfolder
    print("\n🔍 Finding images in YOLO structure...")
    
    all_images = []
    
    # YOLO format: train/images/, valid/images/, test/images/
    yolo_dirs = [
        raw_dir / 'train' / 'images',
        raw_dir / 'valid' / 'images', 
        raw_dir / 'test' / 'images',
        # Also check without images subfolder (backup)
        raw_dir / 'train',
        raw_dir / 'valid',
        raw_dir / 'test',
    ]
    
    for img_dir in yolo_dirs:
        if img_dir.exists() and img_dir.is_dir():
            # Find all jpg and png files
            images = list(img_dir.glob('*.jpg')) + \
                     list(img_dir.glob('*.jpeg')) + \
                     list(img_dir.glob('*.png')) + \
                     list(img_dir.glob('*.JPG'))
            
            if images:
                all_images.extend(images)
                print(f"  ✅ Found {len(images)} images in {img_dir.relative_to(raw_dir)}")
    
    # Remove duplicates (in case we found same images in multiple locations)
    all_images = list(set(all_images))
    
    print(f"\n📊 Total unique images found: {len(all_images)}")
    
    if len(all_images) == 0:
        print("\n❌ No images found!")
        print("\n🔍 Searching everywhere in raw_data:")
        for item in raw_dir.rglob('*'):
            if item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f"  📷 {item.relative_to(raw_dir)}")
        return False
    
    # Show sample images
    print("\n📷 Sample images:")
    for img in all_images[:5]:
        print(f"  - {img.name}")
    
    # Shuffle all images
    random.seed(42)
    random.shuffle(all_images)
    
    # Distribute to 5 grades (20% each)
    print("\n📊 Distributing to grades...")
    n = len(all_images)
    
    grade_distribution = {
        'grade_0': all_images[:int(n*0.20)],           # 20% - Healthy/Normal
        'grade_1': all_images[int(n*0.20):int(n*0.40)],  # 20% - Doubtful
        'grade_2': all_images[int(n*0.40):int(n*0.60)],  # 20% - Minimal
        'grade_3': all_images[int(n*0.60):int(n*0.80)],  # 20% - Moderate
        'grade_4': all_images[int(n*0.80):],           # 20% - Severe
    }
    
    # Copy to train/val/test splits
    print("\n📦 Organizing files...")
    stats = {}
    
    for grade, images in grade_distribution.items():
        n_grade = len(images)
        
        # Split: 70% train, 15% val, 15% test
        train_idx = int(n_grade * 0.70)
        val_idx = int(n_grade * 0.85)
        
        train_imgs = images[:train_idx]
        val_imgs = images[train_idx:val_idx]
        test_imgs = images[val_idx:]
        
        # Copy files
        print(f"\n  {grade}:")
        
        for i, img in enumerate(train_imgs):
            dest = output_dir / 'train' / grade / f"{grade}_train_{i:04d}{img.suffix}"
            shutil.copy2(img, dest)
        print(f"    ✅ Train: {len(train_imgs)}")
        
        for i, img in enumerate(val_imgs):
            dest = output_dir / 'val' / grade / f"{grade}_val_{i:04d}{img.suffix}"
            shutil.copy2(img, dest)
        print(f"    ✅ Val: {len(val_imgs)}")
        
        for i, img in enumerate(test_imgs):
            dest = output_dir / 'test' / grade / f"{grade}_test_{i:04d}{img.suffix}"
            shutil.copy2(img, dest)
        print(f"    ✅ Test: {len(test_imgs)}")
        
        stats[grade] = {
            'train': len(train_imgs),
            'val': len(val_imgs),
            'test': len(test_imgs)
        }
    
    # Print summary
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE!")
    print("="*60)
    
    print(f"\n{'Grade':<12} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-"*60)
    
    for grade, counts in stats.items():
        total = counts['train'] + counts['val'] + counts['test']
        print(f"{grade:<12} {counts['train']:<10} {counts['val']:<10} {counts['test']:<10} {total:<10}")
    
    total_train = sum(s['train'] for s in stats.values())
    total_val = sum(s['val'] for s in stats.values())
    total_test = sum(s['test'] for s in stats.values())
    
    print("-"*60)
    print(f"{'TOTAL':<12} {total_train:<10} {total_val:<10} {total_test:<10} {total_train+total_val+total_test:<10}")
    print("="*60)
    
    print("\n⚠️  IMPORTANT:")
    print("- Images auto-distributed (not medically graded)")
    print("- For demo/testing purposes only")
    print("- Not for clinical diagnosis")
    
    print("\n✅ Dataset ready at: dataset/")
    print("\n🚀 Next steps:")
    print("1. Verify: import os; print(os.listdir('dataset/train/'))")
    print("2. Train: python train_model.py --model xception --epochs 20")
    print("3. Run app: streamlit run app/app.py")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = convert_scoliosis_yolo_dataset()
    
    if not success:
        print("\n💡 TIP: Make sure raw_data has this structure:")
        print("raw_data/")
        print("├── train/images/  ← .jpg files here")
        print("├── valid/images/  ← .jpg files here")
        print("└── test/images/   ← .jpg files here")
