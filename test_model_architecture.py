#!/usr/bin/env python3
"""
Test script to debug model architecture issues
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def create_test_model():
    """Create a simplified test model"""
    # Geometry input
    geom_input = keras.Input(shape=(60, 40, 30, 1), name='geometry')
    
    # Simple encoder
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(geom_input)
    x = layers.MaxPooling3D((2, 2, 2))(x)  # (30, 20, 15)
     
    x = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)  # (15, 10, 7)
    
    # Parameters input
    param_input = keras.Input(shape=(4,), name='parameters')
    param_features = layers.Dense(128, activation='relu')(param_input)
    
    # Flatten geometry features
    geom_features = layers.Flatten()(x)
    geom_features = layers.Dense(512, activation='relu')(geom_features)
    
    # Fusion
    fused = layers.Concatenate()([geom_features, param_features])
    fused = layers.Dense(1024, activation='relu')(fused)
    
    # Reshape for decoder
    fused = layers.Dense(15 * 10 * 7 * 128, activation='relu')(fused)
    fused = layers.Reshape((15, 10, 7, 128))(fused)
    
    # Simple decoder to get back to (60, 40, 30)
    x = layers.Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same')(fused)
    x = layers.UpSampling3D((2, 2, 2))(x)  # (30, 20, 14)
    
    x = layers.Conv3DTranspose(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D((2, 2, 2))(x)  # (60, 40, 28)
    
    # Add padding to z dimension to get exactly 30
    x = layers.ZeroPadding3D(padding=((0, 0), (0, 0), (1, 1)))(x)  # (60, 40, 30)
    
    # Output layers
    velocity_output = layers.Conv3D(3, (1, 1, 1), activation='linear', name='velocity')(x)
    density_output = layers.Conv3D(1, (1, 1, 1), activation='linear', name='density')(x)
    
    model = keras.Model(
        inputs=[geom_input, param_input],
        outputs=[velocity_output, density_output]
    )
    
    return model

def test_model():
    """Test the model with sample data"""
    print("Creating test model...")
    model = create_test_model()
    
    print("Model summary:")
    model.summary()
    
    print("\nTesting with sample data...")
    # Create sample data
    batch_size = 2
    geom_data = np.random.random((batch_size, 60, 40, 30, 1)).astype(np.float32)
    param_data = np.random.random((batch_size, 4)).astype(np.float32)
    
    print(f"Input shapes:")
    print(f"  Geometry: {geom_data.shape}")
    print(f"  Parameters: {param_data.shape}")
    
    # Test prediction
    outputs = model([geom_data, param_data])
    
    print(f"\nOutput shapes:")
    for i, output in enumerate(outputs):
        print(f"  Output {i}: {output.shape}")
    
    print("\n Model architecture test successful!")

if __name__ == "__main__":
    test_model()