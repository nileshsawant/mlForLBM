#!/usr/bin/env python3
"""
Quick test of the efficient model architectures
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_lbm_cnn_model(input_shape_geom=(60, 40, 30, 1), input_shape_params=(4,)):
    """
    Create an efficient 3D CNN model for LBM flow prediction
    
    Inspired by CNO and efficient CNN architectures:
    - Much smaller model (~5-10M parameters vs 173M)
    - Separable convolutions for efficiency
    - Residual connections for better training
    - Fewer but wider layers
    """
    # Geometry input branch (efficient 3D CNN encoder)
    geom_input = keras.Input(shape=input_shape_geom, name='geometry')
    
    # Lightweight encoder with separable convolutions
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(geom_input)
    shortcut1 = x
    x = layers.SeparableConv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)  # (30, 20, 15)
    
    x = layers.SeparableConv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    shortcut2 = x
    x = layers.SeparableConv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)  # (15, 10, 7)
    
    # Compact feature extraction
    geom_features = layers.GlobalAveragePooling3D()(x)  # Much more efficient than Flatten
    geom_features = layers.Dense(128, activation='relu')(geom_features)
    
    # Parameters input branch (compact MLP)
    param_input = keras.Input(shape=input_shape_params, name='parameters')
    param_features = layers.Dense(32, activation='relu')(param_input)
    param_features = layers.Dense(64, activation='relu')(param_features)
    
    # Efficient fusion
    fused = layers.Concatenate()([geom_features, param_features])
    fused = layers.Dense(256, activation='relu')(fused)
    
    # Reshape for efficient decoder
    fused = layers.Dense(15 * 10 * 7 * 32, activation='relu')(fused)  # Much smaller
    fused = layers.Reshape((15, 10, 7, 32))(fused)
    
    # Efficient decoder with skip connections
    x = layers.Conv3DTranspose(64, (3, 3, 3), activation='relu', padding='same')(fused)
    x = layers.UpSampling3D((2, 2, 2))(x)  # (30, 20, 14)
    
    # Add residual connection (resize shortcut2 to match)
    shortcut2_up = layers.UpSampling3D((2, 2, 1))(shortcut2)  # (30, 20, 14)
    x = layers.Add()([x, shortcut2_up])  # Residual connection
    
    x = layers.SeparableConv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling3D((2, 2, 2))(x)  # (60, 40, 28)
    
    # Final adjustment to (60, 40, 30)
    x = layers.ZeroPadding3D(padding=((0, 0), (0, 0), (1, 1)))(x)  # (60, 40, 30)
    
    # Add another residual connection (resize shortcut1)
    shortcut1_up = layers.UpSampling3D((2, 2, 1))(shortcut1)  # (60, 40, 30)
    # Adjust channels to match
    shortcut1_up = layers.Conv3D(32, (1, 1, 1), padding='same')(shortcut1_up)
    x = layers.Add()([x, shortcut1_up])  # Residual connection
    
    # Final feature refinement (shared backbone)
    shared_features = layers.SeparableConv3D(16, (3, 3, 3), activation='relu', padding='same')(x)
    
    # Output branches for different LBM fields (much more efficient)
    velocity_output = layers.Conv3D(3, (1, 1, 1), activation='linear', name='velocity')(shared_features)
    heat_flux_output = layers.Conv3D(3, (1, 1, 1), activation='linear', name='heat_flux')(shared_features)
    density_output = layers.Conv3D(1, (1, 1, 1), activation='linear', name='density')(shared_features)
    energy_output = layers.Conv3D(1, (1, 1, 1), activation='linear', name='energy')(shared_features)
    temperature_output = layers.Conv3D(1, (1, 1, 1), activation='linear', name='temperature')(shared_features)
    
    # Create model
    model = keras.Model(
        inputs=[geom_input, param_input],
        outputs=[velocity_output, heat_flux_output, density_output, energy_output, temperature_output]
    )
    
    return model

def create_simple_efficient_model(input_shape_geom=(60, 40, 30, 1), input_shape_params=(4,)):
    """
    Even simpler model focusing on minimal parameters
    """
    # Inputs
    geom_input = keras.Input(shape=input_shape_geom, name='geometry')
    param_input = keras.Input(shape=input_shape_params, name='parameters')
    
    # Very simple geometry encoder - just global pooling
    geom_features = layers.GlobalAveragePooling3D()(geom_input)
    geom_features = layers.Dense(64, activation='relu')(geom_features)
    
    # Simple parameter encoder
    param_features = layers.Dense(32, activation='relu')(param_input)
    
    # Fusion
    fused = layers.Concatenate()([geom_features, param_features])
    fused = layers.Dense(128, activation='relu')(fused)
    fused = layers.Dense(256, activation='relu')(fused)
    
    # Reshape and broadcast to full geometry
    fused = layers.Dense(32, activation='relu')(fused)
    fused = layers.Reshape((1, 1, 1, 32))(fused)
    fused = tf.tile(fused, [1, 60, 40, 30, 1])  # Broadcast
    
    # Simple decoder
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(fused)
    
    # Outputs
    velocity_output = layers.Conv3D(3, (1, 1, 1), activation='linear', name='velocity')(x)
    heat_flux_output = layers.Conv3D(3, (1, 1, 1), activation='linear', name='heat_flux')(x)
    density_output = layers.Conv3D(1, (1, 1, 1), activation='linear', name='density')(x)
    energy_output = layers.Conv3D(1, (1, 1, 1), activation='linear', name='energy')(x)
    temperature_output = layers.Conv3D(1, (1, 1, 1), activation='linear', name='temperature')(x)
    
    model = keras.Model(
        inputs=[geom_input, param_input],
        outputs=[velocity_output, heat_flux_output, density_output, energy_output, temperature_output]
    )
    
    return model

def main():
    print("Testing Efficient Model Architectures")
    print("=" * 50)
    
    try:
        # Test Model 1: Efficient CNN with separable convolutions
        print("\n1. Testing Efficient CNN with Separable Convolutions...")
        model1 = create_lbm_cnn_model()
        param_count_1 = model1.count_params()
        model_size_1 = param_count_1 * 4 / (1024**2)  # MB
        print(f"    Success! Parameters: {param_count_1:,} ({model_size_1:.1f} MB)")
        
        # Test forward pass
        geom_test = np.random.random((1, 60, 40, 30, 1)).astype(np.float32)
        param_test = np.random.random((1, 4)).astype(np.float32)
        outputs1 = model1([geom_test, param_test])
        print(f"    Forward pass successful! Output shapes:")
        for i, out in enumerate(outputs1):
            print(f"      Output {i+1}: {out.shape}")
            
    except Exception as e:
        print(f"    Model 1 failed: {e}")
    
    try:
        # Test Model 2: Ultra-simple model
        print("\n2. Testing Ultra-Simple Model...")
        model2 = create_simple_efficient_model()
        param_count_2 = model2.count_params()
        model_size_2 = param_count_2 * 4 / (1024**2)  # MB
        print(f"    Success! Parameters: {param_count_2:,} ({model_size_2:.1f} MB)")
        
        # Test forward pass
        outputs2 = model2([geom_test, param_test])
        print(f"    Forward pass successful! Output shapes:")
        for i, out in enumerate(outputs2):
            print(f"      Output {i+1}: {out.shape}")
            
    except Exception as e:
        print(f"    Model 2 failed: {e}")
    
    print(f"\n Comparison with original 173M parameter model:")
    if 'param_count_1' in locals():
        print(f"   Model 1 reduction: {173_000_000 / param_count_1:.1f}x smaller")
    if 'param_count_2' in locals():
        print(f"   Model 2 reduction: {173_000_000 / param_count_2:.1f}x smaller")

if __name__ == "__main__":
    main()