#!/usr/bin/env python3
"""
Debug script to examine the structure of the neural network prediction .npz files
"""

import numpy as np
import os

def inspect_npz_file(npz_file):
    """Inspect the structure of an NPZ file"""
    print(f" Inspecting: {npz_file}")
    print("=" * 80)
    
    data = np.load(npz_file, allow_pickle=True)
    
    print(" Available keys:")
    for key in data.keys():
        print(f"   {key}")
    
    print("\n Data shapes and types:")
    for key in data.keys():
        arr = data[key]
        print(f"   {key}: shape={arr.shape}, dtype={arr.dtype}")
        if len(arr.shape) > 0 and arr.shape[0] <= 5:
            print(f"      First few values: {arr[:3] if len(arr.shape)==1 else 'array'}")
    
    print("\n" + "=" * 80)

def main():
    npz_dir = "validation/neural_network_predictions"
    
    if not os.path.exists(npz_dir):
        print(f" Directory not found: {npz_dir}")
        return
    
    npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
    
    if not npz_files:
        print(f" No .npz files found in {npz_dir}")
        return
    
    print(f" Found {len(npz_files)} NPZ files")
    print()
    
    # Inspect the first file in detail
    inspect_npz_file(npz_files[0])

if __name__ == "__main__":
    main()