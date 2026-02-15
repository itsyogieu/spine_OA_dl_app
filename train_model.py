"""
Simple Spine X-ray Classification Model Training Script

This script provides a quick way to train a model if you don't want to use notebooks.

Usage:
    python train_model.py --epochs 50 --batch-size 32 --model xception
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception, ResNet50, InceptionResNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from datetime import datetime

def create_model(model_name='xception', num_classes=5, img_size=(224, 224)):
    """Create transfer learning model"""
    
    # Select base model
    if model_name == 'xception':
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    elif model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    elif model_name == 'inception_resnet_v2':
        base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_average_pooling2d_1')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=outputs)
    
    return model, base_model

def train_model(args):
    """Main training function"""
    
    print("="*60)
    print("SPINE X-RAY CLASSIFICATION MODEL TRAINING")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("="*60)
    
    # Setup directories
    train_dir = Path(args.data_dir) / 'train'
    val_dir = Path(args.data_dir) / 'val'
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1]
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    img_size = (args.img_size, args.img_size)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"\n✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Number of classes: {train_generator.num_classes}")
    
    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    
    class_counts = np.bincount(train_generator.classes)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(class_counts)),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    print("\n📊 Class weights:")
    for i, w in class_weight_dict.items():
        print(f"  Class {i}: {w:.3f}")
    
    # Create model
    print(f"\n🔨 Building {args.model} model...")
    model, base_model = create_model(args.model, train_generator.num_classes, img_size)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created with {model.count_params():,} parameters")
    
    # Callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = model_dir / f"model_{args.model}_{timestamp}.hdf5"
    
    callbacks = [
        ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Phase 1: Train only the top layers
    print("\n" + "="*60)
    print("PHASE 1: Training top layers only")
    print("="*60)
    
    history1 = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=args.epochs // 2,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune entire model
    if args.fine_tune:
        print("\n" + "="*60)
        print("PHASE 2: Fine-tuning entire model")
        print("="*60)
        
        # Unfreeze base model
        base_model.trainable = True
        
        # Recompile with lower learning rate
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        history2 = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=args.epochs // 2,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
    
    # Save final model
    final_model_path = model_dir / f"model_{args.model}_spine_ft.hdf5"
    model.save(final_model_path)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Model saved to: {final_model_path}")
    print(f"📊 Best validation accuracy: {max(history1.history['val_accuracy']):.4f}")
    print("="*60)
    
    return model, history1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train spine X-ray classification model")
    parser.add_argument('--model', type=str, default='xception',
                       choices=['xception', 'resnet50', 'inception_resnet_v2'],
                       help='Model architecture to use')
    parser.add_argument('--data-dir', type=str, default='dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model-dir', type=str, default='src/models',
                       help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=224,
                       help='Image size (will be square)')
    parser.add_argument('--fine-tune', action='store_true',
                       help='Perform fine-tuning after initial training')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.data_dir).exists():
        print(f"❌ Error: Dataset directory not found: {args.data_dir}")
        print("Please prepare your dataset first. See DATASET_INFO.md for details.")
        exit(1)
    
    # Train model
    model, history = train_model(args)
    
    print("\n🎉 All done! You can now use the model in the Streamlit app.")
    print(f"   Copy the model to: src/models/model_Xception_spine_ft.hdf5")
    print(f"   Then run: streamlit run app/app.py")
