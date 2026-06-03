# src/models/cnn_model.py
# PURPOSE: CNN model for flood/no-flood image classification

import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# STEP 1 — Generate Synthetic Training Data
# ─────────────────────────────────────────
def generate_flood_images(num_samples=2000, img_size=64):
    """
    Generate synthetic satellite-like images for training.
    
    Since downloading thousands of real satellite images
    requires huge storage, we generate realistic synthetic
    images that mimic flood and non-flood patterns.
    
    Real project extension: Replace with actual Sentinel-2
    image patches downloaded via GEE.
    
    Returns:
        X: Image array (num_samples, img_size, img_size, 3)
        y: Labels array (0=No Flood, 1=Flood)
    """
    print(f"\n🖼️ Generating {num_samples} synthetic satellite images...")
    
    X = []  # Images
    y = []  # Labels
    
    half = num_samples // 2  # Half flood, half no-flood
    
    # Generate NO FLOOD images (green/brown tones)
    for i in range(half):
        img = np.zeros((img_size, img_size, 3))
        
        # Green channel dominant = vegetation
        img[:, :, 0] = np.random.uniform(0.1, 0.3,  # Red channel
                        (img_size, img_size))
        img[:, :, 1] = np.random.uniform(0.4, 0.8,  # Green channel
                        (img_size, img_size))
        img[:, :, 2] = np.random.uniform(0.1, 0.3,  # Blue channel
                        (img_size, img_size))
        
        # Add some texture noise
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        X.append(img)
        y.append(0)  # Label: No Flood
    
    # Generate FLOOD images (blue/dark tones)
    for i in range(half):
        img = np.zeros((img_size, img_size, 3))
        
        # Blue channel dominant = water
        img[:, :, 0] = np.random.uniform(0.05, 0.2,  # Red channel
                        (img_size, img_size))
        img[:, :, 1] = np.random.uniform(0.1, 0.3,   # Green channel
                        (img_size, img_size))
        img[:, :, 2] = np.random.uniform(0.4, 0.9,   # Blue channel
                        (img_size, img_size))
        
        # Add flood patches (darker areas)
        patch_x = np.random.randint(0, img_size//2)
        patch_y = np.random.randint(0, img_size//2)
        patch_size = np.random.randint(10, 30)
        
        img[patch_x:patch_x+patch_size,
            patch_y:patch_y+patch_size, :] *= 0.3
        
        # Add texture noise
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)
        
        X.append(img)
        y.append(1)  # Label: Flood
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"   ✅ Generated {len(X)} images")
    print(f"   Image shape: {X.shape}")
    print(f"   Flood images: {sum(y==1)}")
    print(f"   No-flood images: {sum(y==0)}")
    
    return X, y

# ─────────────────────────────────────────
# STEP 2 — Build CNN Architecture
# ─────────────────────────────────────────
def build_cnn_model(img_size=64):
    """
    Build CNN model architecture.
    
    CNN Architecture:
    Input Image (64x64x3)
         ↓
    Conv2D + ReLU    (extract edges)
         ↓
    MaxPooling2D     (reduce size)
         ↓
    Conv2D + ReLU    (extract patterns)
         ↓
    MaxPooling2D     (reduce size)
         ↓
    Conv2D + ReLU    (extract features)
         ↓
    GlobalAveragePooling2D
         ↓
    Dense + Dropout  (classify)
         ↓
    Output (Flood/No Flood)
    """
    print("\n🧠 Building CNN Architecture...")
    
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(img_size, img_size, 3)),
        
        # Block 1 — Edge detection
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Block 2 — Pattern recognition
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Block 3 — Feature extraction
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        
        # Classification head
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary output
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',  # Binary classification loss
        metrics=['accuracy']
    )
    
    print("   ✅ CNN model built!")
    print(f"\n📋 Model Summary:")
    model.summary()
    
    return model

# ─────────────────────────────────────────
# STEP 3 — Train CNN Model
# ─────────────────────────────────────────
def train_cnn(model, X_train, y_train, X_val, y_val):
    """
    Train the CNN model with callbacks.
    
    Callbacks:
    - EarlyStopping: Stop if no improvement
    - ModelCheckpoint: Save best model
    - ReduceLROnPlateau: Reduce learning rate if stuck
    """
    print("\n🏋️ Training CNN model...")
    
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        # Stop training if val_loss doesn't improve for 5 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Save best model automatically
        ModelCheckpoint(
            'models/cnn_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,          # Maximum 30 epochs
        batch_size=32,      # 32 images per batch
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n✅ CNN training complete!")
    return history

# ─────────────────────────────────────────
# STEP 4 — Plot Training History
# ─────────────────────────────────────────
def plot_training_history(history):
    """Plot accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'],
                 label='Train Accuracy', color='steelblue')
    axes[0].plot(history.history['val_accuracy'],
                 label='Val Accuracy', color='orange')
    axes[0].set_title('CNN Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'],
                 label='Train Loss', color='steelblue')
    axes[1].plot(history.history['val_loss'],
                 label='Val Loss', color='orange')
    axes[1].set_title('CNN Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('reports/figures/cnn_training_history.png', dpi=150)
    plt.show()
    print("✅ Training history plot saved!")

# ─────────────────────────────────────────
# STEP 5 — Evaluate and Visualize
# ─────────────────────────────────────────
def evaluate_cnn(model, X_test, y_test):
    """Evaluate CNN on test data."""
    print("\n📊 Evaluating CNN model...")
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Test Loss:     {loss:.4f}")
    
    return accuracy

def visualize_predictions(model, X_test, y_test, num_samples=10):
    """Show sample predictions."""
    predictions = model.predict(X_test[:num_samples], verbose=0)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(X_test[i])
        
        pred = predictions[i][0]
        actual = y_test[i]
        
        pred_label = "🌊 Flood" if pred > 0.5 else "✅ No Flood"
        actual_label = "Flood" if actual == 1 else "No Flood"
        
        color = 'green' if (pred > 0.5) == actual else 'red'
        
        axes[i].set_title(
            f"Pred: {pred_label}\nActual: {actual_label}",
            color=color, fontsize=8
        )
        axes[i].axis('off')
    
    plt.suptitle('CNN Predictions vs Actual', fontsize=14)
    plt.tight_layout()
    plt.savefig('reports/figures/cnn_predictions.png', dpi=150)
    plt.show()
    print("✅ Prediction visualization saved!")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    print("="*50)
    print("  FLOOD CNN — DEEP LEARNING MODEL")
    print("="*50)
    
    # Generate data
    X, y = generate_flood_images(num_samples=3000)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {len(X_train)}")
    print(f"   Val:   {len(X_val)}")
    print(f"   Test:  {len(X_test)}")
    
    # Build model
    model = build_cnn_model(img_size=64)
    
    # Train model
    history = train_cnn(model, X_train, y_train, X_val, y_val)
    
    # Evaluate
    accuracy = evaluate_cnn(model, X_test, y_test)
    
    # Plot history
    plot_training_history(history)
    
    # Visualize predictions
    visualize_predictions(model, X_test, y_test)
    
    print("\n" + "="*50)
    print("🎉 PHASE 6 COMPLETE!")
    print(f"   CNN Accuracy: {accuracy*100:.2f}%")
    print("   Model saved: models/cnn_model.keras")
    print("="*50)