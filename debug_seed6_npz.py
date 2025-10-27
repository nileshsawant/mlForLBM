#!/usr/bin/env python3
"""
Debug script to check the structure of seed6 validation .npz files
"""
import numpy as np

# Check one of the seed6 validation files
npz_file = "validation_seed6/neural_network_predictions/validation_case_000_nu_0.00259_temp_0.025_geom_6_nn_predictions.npz"

print("Checking seed6 validation .npz file structure...")
print(f"File: {npz_file}")
print("=" * 60)

try:
    data = np.load(npz_file)
    
    print("Available keys:")
    for key in data.files:
        print(f"  {key}: shape = {data[key].shape}")
        
    print(f"\nTotal keys: {len(data.files)}")
    
except Exception as e:
    print(f"Error loading file: {e}")