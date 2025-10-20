#!/usr/bin/env python3
"""
Neural Network Training Script for LBM Flow Prediction - Enhanced Version

This script trains a 3D CNN to predict flow fields in cracked geometries.
CHANGES FROM ORIGINAL:
1. Removed is_fluid masking - trains on all spatial points
2. Added adaptive learning rate scheduling

Input: geometry (60x40x30), parameters (nu, alpha, temp, time)
Output: flow fields (velocity, pressure, temperature) on all points
"""

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_ml_training_data(data_dir="ml_training_data", max_cases=None, sample_timesteps=None):
    """
    Load processed ML training data
    
    Parameters:
    -----------
    data_dir : str
        Directory containing processed ML data
    max_cases : int, optional
        Maximum number of cases to load (for testing)
    sample_timesteps : int, optional
        Sample every N timesteps (e.g., 5 = every 50 time units)
        
    Returns:
    --------
    dict : Training data with inputs, outputs, and masks
    """
    print("Loading ML Training Data")
    print("=" * 50)
    
    # Find all case directories
    case_pattern = os.path.join(data_dir, "case_*")
    case_dirs = sorted(glob.glob(case_pattern))
    
    if max_cases:
        case_dirs = case_dirs[:max_cases]
        print(f"Limited to first {max_cases} cases for testing")
    
    print(f"Loading data from {len(case_dirs)} cases...")
    
    # Initialize lists to store data
    geometries = []
    parameters = []
    velocity_fields = []
    heat_flux_fields = []
    density_fields = []
    energy_fields = []
    temperature_fields = []
    
    total_samples = 0
    
    for i, case_dir in enumerate(case_dirs):
        case_name = os.path.basename(case_dir)
        
        # Load metadata
        metadata_file = os.path.join(case_dir, f"{case_name}_metadata.json")
        if not os.path.exists(metadata_file):
            print(f"  Skipping {case_name}: no metadata file")
            continue
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        case_info = metadata['case_info']
        nu_val = case_info['nu_value']
        temp_val = case_info['temp_value'] 
        alpha_val = case_info['alpha_value']
        geom_id = case_info['geom_id']
        
        # Load geometry (same for all timesteps in this case)
        geometry_file = f"microstructure_geom_{geom_id}.csv"
        # Geometry files are in the root directory, not the parent of ml_training_data
        geometry_path = os.path.join("/Users/nsawant/apps/nileshsawant/mlForLBM", geometry_file)
        
        if os.path.exists(geometry_path):
            # Load geometry as 3D array (assuming CSV format: x,y,z,value)
            import pandas as pd
            geom_df = pd.read_csv(geometry_path)
            
            # Reshape to 3D grid (60x40x30)
            geometry_3d = np.zeros((60, 40, 30))
            for _, row in geom_df.iterrows():
                # Columns are X, Y, Z, tag (1=solid, 0=fluid)
                x, y, z = int(row['X']), int(row['Y']), int(row['Z'])
                if x < 60 and y < 40 and z < 30:
                    geometry_3d[x, y, z] = row['tag']  # solid/fluid indicator
        else:
            print(f"  Warning: Geometry file not found: {geometry_path}")
            continue
        
        # Load NPZ files for this case
        npz_pattern = os.path.join(case_dir, "plt*_ml.npz")
        npz_files = sorted(glob.glob(npz_pattern))
        
        if sample_timesteps:
            npz_files = npz_files[::sample_timesteps]
        
        case_samples = 0
        for npz_file in npz_files:
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                # Extract actual simulation time from filename
                # Files are named plt00000, plt00010, plt00020, ..., plt01000
                # This corresponds to timesteps 0, 10, 20, ..., 1000
                filename = os.path.basename(npz_file)
                plt_number = filename.split('_')[0]  # Extract 'plt00010' part
                timestep = int(plt_number[3:])       # Extract '00010' -> 10
                time_val = float(timestep) / 1000.0  # Normalize to 0.0-1.0 for neural network
                
                # Check if required fields exist (removed is_fluid from required)
                required_fields = ['rho', 'vel_x', 'vel_y', 'vel_z', 'qx', 'qy', 'qz', 'two_rho_e', 'temperature']
                missing_fields = [field for field in required_fields if field not in data.keys()]
                
                if missing_fields:
                    # Find available field names for debugging
                    available_fields = list(data.keys())
                    print(f"    Missing fields: {missing_fields}")
                    print(f"    Available fields: {available_fields[:10]}...")  # Show first 10
                    
                    # Skip this file if required fields are missing
                    continue
                
                # Load data arrays from NPZ file (ensure float32 for neural network)
                rho = np.array(data['rho']).astype(np.float32)
                velx = np.array(data['vel_x']).astype(np.float32)
                vely = np.array(data['vel_y']).astype(np.float32)
                velz = np.array(data['vel_z']).astype(np.float32)
                qx = np.array(data['qx']).astype(np.float32)
                qy = np.array(data['qy']).astype(np.float32)
                qz = np.array(data['qz']).astype(np.float32)
                two_rho_e = np.array(data['two_rho_e']).astype(np.float32)
                temperature = np.array(data['temperature']).astype(np.float32)
                
                # Store all field data
                geometries.append(geometry_3d)
                # Parameters: [viscosity, thermal_diffusivity, body_temperature, time]
                # Time normalized to 0.0 (initial) to 1.0 (final) for transient simulations
                parameters.append([nu_val, alpha_val, temp_val, time_val])
                
                # Stack velocity components: (60,40,30,3)
                velocity_fields.append(np.stack([velx, vely, velz], axis=-1))
                
                # Stack heat flux components: (60,40,30,3)
                heat_flux_fields.append(np.stack([qx, qy, qz], axis=-1))
                
                # Store scalar fields: (60,40,30,1)
                density_fields.append(rho[..., np.newaxis])
                energy_fields.append(two_rho_e[..., np.newaxis])
                temperature_fields.append(temperature[..., np.newaxis])
                
                case_samples += 1
                    
            except Exception as e:
                print(f"    Error loading {npz_file}: {e}")
                continue
        
        total_samples += case_samples
        print(f"  {case_name}: Loaded {case_samples} timesteps")
        
        # Progress update
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i+1}/{len(case_dirs)} cases, {total_samples} total samples")
    
    print(f"\nLoaded {total_samples} training samples from {len(case_dirs)} cases")
    
    # Convert to numpy arrays
    training_data = {
        'geometries': np.array(geometries),           # (N, 60, 40, 30)
        'parameters': np.array(parameters),           # (N, 4) [viscosity, thermal_diffusivity, body_temp, time(0.0-1.0)]
        'velocity_fields': np.array(velocity_fields), # (N, 60, 40, 30, 3) [velx, vely, velz]
        'heat_flux_fields': np.array(heat_flux_fields), # (N, 60, 40, 30, 3) [qx, qy, qz] 
        'density_fields': np.array(density_fields),   # (N, 60, 40, 30)
        'energy_fields': np.array(energy_fields),     # (N, 60, 40, 30)
        'temperature_fields': np.array(temperature_fields) # (N, 60, 40, 30)
    }
    
    return training_data

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
    
    # Lightweight encoder with small channel counts
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(geom_input)
    shortcut1 = x
    x = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same')(x)  # Much smaller than 32
    x = layers.MaxPooling3D((2, 2, 2))(x)  # (30, 20, 15)
    
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)  # Much smaller than 64
    shortcut2 = x
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)  # Keep same size
    x = layers.MaxPooling3D((2, 2, 2))(x)  # (15, 10, 7)
    
    # Compact feature extraction
    geom_features = layers.GlobalAveragePooling3D()(x)  # Much more efficient than Flatten
    geom_features = layers.Dense(64, activation='relu')(geom_features)  # Smaller
    
    # Parameters input branch (compact MLP)
    param_input = keras.Input(shape=input_shape_params, name='parameters')
    param_features = layers.Dense(16, activation='relu')(param_input)  # Much smaller
    param_features = layers.Dense(32, activation='relu')(param_features)  # Much smaller
    
    # Efficient fusion
    fused = layers.Concatenate()([geom_features, param_features])
    fused = layers.Dense(128, activation='relu')(fused)  # Much smaller than 256
    
    # Reshape for efficient decoder
    fused = layers.Dense(15 * 10 * 7 * 16, activation='relu')(fused)  # Even smaller
    fused = layers.Reshape((15, 10, 7, 16))(fused)
    
    # Efficient decoder - simplified approach without problematic residual connections
    x = layers.Conv3DTranspose(32, (3, 3, 3), activation='relu', padding='same')(fused)  # Smaller
    x = layers.UpSampling3D((2, 2, 2))(x)  # (30, 20, 14)
    
    x = layers.Conv3D(24, (3, 3, 3), activation='relu', padding='same')(x)  # Regular Conv3D
    x = layers.UpSampling3D((2, 2, 2))(x)  # (60, 40, 28)
    
    # Final adjustment to (60, 40, 30)
    x = layers.ZeroPadding3D(padding=((0, 0), (0, 0), (1, 1)))(x)  # (60, 40, 30)
    
    # Final feature refinement (shared backbone)
    shared_features = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(x)  # Regular Conv3D
    
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

def create_ultra_efficient_cno_model(input_shape_geom=(60, 40, 30, 1), input_shape_params=(4,)):
    """
    Ultra-efficient CNO-inspired model for LBM flow prediction
    
    Based on Course Tutorial 06 - CNO architecture:
    - Lift/Project operations
    - 1D convolutions along different dimensions  
    - Minimal parameters (~1-2M total)
    - Separable convolutions throughout
    """
    # Inputs
    geom_input = keras.Input(shape=input_shape_geom, name='geometry')
    param_input = keras.Input(shape=input_shape_params, name='parameters')
    
    # Lift operation: project geometry to feature space
    lifted = layers.Conv3D(16, (1, 1, 1), activation='relu', padding='same')(geom_input)
    
    # Parameter embedding - simpler approach
    param_emb = layers.Dense(16, activation='relu')(param_input)
    param_emb = layers.RepeatVector(60*40*30)(param_emb)  # Repeat for each voxel
    param_emb = layers.Reshape((60, 40, 30, 16))(param_emb)  # Match geometry shape
    
    # Fusion via addition (more efficient than concatenation)
    x = layers.Add()([lifted, param_emb])
    
    # Simplified efficient blocks - avoid complex transpose operations
    # Process with regular 3D convolutions but very small kernels and channels
    x = layers.Conv3D(24, (1, 1, 1), activation='relu', padding='same')(x)  # 1x1x1 conv
    x = layers.Conv3D(16, (3, 1, 1), activation='relu', padding='same')(x)  # Process X dimension
    x = layers.Conv3D(16, (1, 3, 1), activation='relu', padding='same')(x)  # Process Y dimension  
    x = layers.Conv3D(16, (1, 1, 3), activation='relu', padding='same')(x)  # Process Z dimension
    x = layers.Conv3D(16, (1, 1, 1), activation='relu', padding='same')(x)  # Final projection
    
    # Project operation: map to output fields
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

def train_lbm_model(training_data, validation_split=0.2, epochs=100, batch_size=4, custom_model=None, use_adaptive=True):
    """
    Train the LBM neural network model with adaptive learning rate (NO MASKING)
    """
    print("Setting up neural network training...")
    
    # Prepare input data
    geometries = training_data['geometries'][..., np.newaxis]  # Add channel dimension
    parameters = training_data['parameters']
    
    # Prepare output data (NO MASKING APPLIED)
    velocities = training_data['velocity_fields']                  # (N, 60, 40, 30, 3) [velx, vely, velz]
    heat_fluxes = training_data['heat_flux_fields']                # (N, 60, 40, 30, 3) [qx, qy, qz]
    densities = training_data['density_fields']                   # (N, 60, 40, 30, 1) [rho]
    energies = training_data['energy_fields']                     # (N, 60, 40, 30, 1) [two_rho_e]
    temperatures = training_data['temperature_fields']            # (N, 60, 40, 30, 1) [T]
    
    # Normalize parameters
    param_scaler = StandardScaler()
    parameters_scaled = param_scaler.fit_transform(parameters)
    
    # Split data
    indices = np.arange(len(geometries))
    train_idx, val_idx = train_test_split(indices, test_size=validation_split, random_state=42)
    
    # Create model
    if custom_model is not None:
        model = custom_model
    else:
        model = create_lbm_cnn_model()
    
    # CHANGE 2: Adaptive learning rate setup
    if use_adaptive:
        # Create adaptive learning rate callbacks
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        callbacks = [reduce_lr, early_stop]
        
        # Compile with standard optimizer (callbacks will handle adaptation)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mae', 'mae', 'mae', 'mae']  # One metric for each of the 5 outputs
        )
        
        # Prepare training and validation data
        X_train = [geometries[train_idx], parameters_scaled[train_idx]]
        y_train = [velocities[train_idx], heat_fluxes[train_idx], densities[train_idx], 
                   energies[train_idx], temperatures[train_idx]]
        
        X_val = [geometries[val_idx], parameters_scaled[val_idx]]
        y_val = [velocities[val_idx], heat_fluxes[val_idx], densities[val_idx], 
                 energies[val_idx], temperatures[val_idx]]
        
        # Train with adaptive learning rate
        print("Training with adaptive learning rate...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, param_scaler, history.history['loss']
    
    else:
        # Original fixed learning rate training (NO MASKING)
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse')
        
        train_losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            # Shuffle training data
            np.random.shuffle(train_idx)
            
            # Mini-batch training (NO MASKING)
            for i in range(0, len(train_idx), batch_size):
                batch_idx = train_idx[i:i+batch_size]
                
                geom_batch = geometries[batch_idx]
                param_batch = parameters_scaled[batch_idx]  
                vel_batch = velocities[batch_idx]
                hf_batch = heat_fluxes[batch_idx]
                dens_batch = densities[batch_idx]
                eng_batch = energies[batch_idx]
                temp_batch = temperatures[batch_idx]
                
                # Train step (NO MASKING - standard MSE loss on all points)
                with tf.GradientTape() as tape:
                    vel_pred, hf_pred, dens_pred, eng_pred, temp_pred = model([geom_batch, param_batch], training=True)
                    
                    # Standard MSE loss on ALL points (no masking)
                    vel_loss = tf.reduce_mean(tf.square(vel_batch - vel_pred))
                    hf_loss = tf.reduce_mean(tf.square(hf_batch - hf_pred))
                    dens_loss = tf.reduce_mean(tf.square(dens_batch - dens_pred))
                    eng_loss = tf.reduce_mean(tf.square(eng_batch - eng_pred))
                    temp_loss = tf.reduce_mean(tf.square(temp_batch - temp_pred))
                    
                    total_loss = vel_loss + hf_loss + dens_loss + eng_loss + temp_loss
                
                # Backward pass
                gradients = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                epoch_losses.append(float(total_loss))
            
            avg_loss = np.mean(epoch_losses)
            train_losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")
        
        return model, param_scaler, train_losses

def main():
    """
    Main training pipeline with model selection
    """
    print("LBM Neural Network Training Pipeline - Enhanced Version")
    print("=" * 60)
    
    # Load full training data (all 125 cases, all timesteps)
    training_data = load_ml_training_data()  # Remove limits for full training
    
    if len(training_data['geometries']) == 0:
        print(" No training data loaded!")
        return
    
    print(f"Training data shapes:")
    for key, data in training_data.items():
        print(f"  {key}: {data.shape}")
    
    # Create and compare different models
    print("\n  Model Architecture Comparison:")
    print("=" * 50)
    
    # Model 1: Efficient CNN (your current optimized version)
    print("1. Efficient 3D CNN (Separable Conv + Residual):")
    model_efficient = create_lbm_cnn_model()
    param_count_1 = model_efficient.count_params()
    model_size_1 = param_count_1 * 4 / (1024**2)  # MB (assuming float32)
    print(f"   Parameters: {param_count_1:,} ({model_size_1:.1f} MB)")
    
    print("\n2. Ultra-Efficient CNO-inspired:")
    model_cno = create_ultra_efficient_cno_model()
    param_count_2 = model_cno.count_params()
    model_size_2 = param_count_2 * 4 / (1024**2)  # MB
    print(f"   Parameters: {param_count_2:,} ({model_size_2:.1f} MB)")
    
    # Choose the most efficient model
    if param_count_2 < param_count_1:
        chosen_model = model_cno
        model_name = "CNO-inspired"
        print(f"\n Using ultra-efficient {model_name} model ({param_count_2:,} parameters)")
    else:
        chosen_model = model_efficient  
        model_name = "Efficient CNN"
        print(f"\n Using {model_name} model ({param_count_1:,} parameters)")
    
    print(f"\n Parameter reduction from original 173M: {173_000_000 / chosen_model.count_params():.1f}x smaller!")
    print("\n ENHANCEMENTS:")
    print("  - Removed is_fluid masking: trains on ALL spatial points")
    print("  - Added adaptive learning rate with ReduceLROnPlateau and EarlyStopping")
    
    # Train the chosen model with adaptive learning rate (reduced epochs based on plateau analysis)
    model, scaler, losses = train_lbm_model(training_data, epochs=100, batch_size=8, 
                                            custom_model=chosen_model, use_adaptive=True)
    
    # Save model
    model_filename = f'lbm_flow_predictor_{model_name.lower().replace(" ", "_")}_enhanced.h5'
    model.save(model_filename)
    print(f" Enhanced model saved as '{model_filename}'")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f'Training Loss - Enhanced {model_name} (No Masking + Adaptive LR)\n({chosen_model.count_params():,} params)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.savefig(f'training_loss_{model_name.lower().replace(" ", "_")}_enhanced.png')
    plt.show()

if __name__ == "__main__":
    main()