#!/usr/bin/env python3
"""
Minimal debug script for training issues
"""

import os
import sys
sys.path.append('/Users/nsawant/apps/nileshsawant/mlForLBM')

from train_lbm_neural_network import load_ml_training_data
import numpy as np

def debug_data_shapes():
    """Debug data loading and shapes"""
    print("Loading minimal training data...")
    training_data = load_ml_training_data(max_cases=1, sample_timesteps=20)
    
    print("\nData shapes after loading:")
    for key, data in training_data.items():
        print(f"  {key}: {data.shape} (dtype: {data.dtype})")
    
    # Test a single batch
    print("\nTesting single batch extraction...")
    batch_size = 2
    n_samples = len(training_data['geometries'])
    
    if n_samples < batch_size:
        print(f"Warning: Only {n_samples} samples available, using batch_size=1")
        batch_size = 1
    
    batch_idx = np.arange(batch_size)
    
    geom_batch = training_data['geometries'][batch_idx]
    param_batch = training_data['parameters'][batch_idx]
    vel_batch = training_data['velocity_fields'][batch_idx]
    dens_batch = training_data['density_fields'][batch_idx]
    mask_batch = training_data['fluid_masks'][batch_idx]
    
    print(f"\nBatch shapes:")
    print(f"  geom_batch: {geom_batch.shape}")
    print(f"  param_batch: {param_batch.shape}")
    print(f"  vel_batch: {vel_batch.shape}")
    print(f"  dens_batch: {dens_batch.shape}")
    print(f"  mask_batch: {mask_batch.shape}")
    
    print("\n Data loading debug complete!")
    return training_data

if __name__ == "__main__":
    debug_data_shapes()