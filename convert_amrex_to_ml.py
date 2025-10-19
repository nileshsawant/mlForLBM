#!/usr/bin/env python3
"""
Convert AMReX plotfile to ML-friendly formats (NumPy, HDF5, CSV)
"""

import yt
import numpy as np
import h5py
import pandas as pd
import os
import sys

def convert_amrex_to_ml_formats(plotfile_path, output_prefix="ml_data"):
    """
    Convert AMReX plotfile to multiple ML-friendly formats
    
    Parameters:
    -----------
    plotfile_path : str
        Path to the AMReX plotfile directory
    output_prefix : str
        Prefix for output files
    """
    
    print(f"Loading AMReX dataset: {plotfile_path}")
    
    # Load the dataset with yt
    ds = yt.load(plotfile_path)
    
    # Print information about the dataset
    print(f"Dataset info:")
    print(f"  Domain: {ds.domain_left_edge} to {ds.domain_right_edge}")
    print(f"  Fields: {ds.field_list}")
    print(f"  Resolution: {ds.domain_dimensions}")
    
    # Create a covering grid (uniform grid covering the entire domain)
    # This is perfect for ML as it gives you regular arrays
    level = 0  # Use the base level
    grid = ds.covering_grid(level=level, 
                           left_edge=ds.domain_left_edge, 
                           dims=ds.domain_dimensions)
    
    # Dictionary to store all field data
    field_data = {}
    field_names = []
    
    # Extract all fields
    for field in ds.field_list:
        field_name = field[1]  # Get the field name (without gas/stream prefix)
        field_names.append(field_name)
        
        print(f"Extracting field: {field_name}")
        data = grid[field].v  # .v gives the raw numpy array without units
        field_data[field_name] = data
        
        print(f"  Shape: {data.shape}, Min: {data.min():.6e}, Max: {data.max():.6e}")
    
    # Get dimensions
    nx, ny, nz = data.shape
    
    # 1. Save as NumPy format (.npz) - Best for direct ML use
    print(f"\nSaving as NumPy format: {output_prefix}.npz")
    np.savez_compressed(f"{output_prefix}.npz", **field_data)
    
    # 2. Save as HDF5 format - Good for large datasets
    print(f"Saving as HDF5 format: {output_prefix}.h5")
    with h5py.File(f"{output_prefix}.h5", 'w') as h5f:
        # Add metadata
        h5f.attrs['domain_left_edge'] = ds.domain_left_edge.v
        h5f.attrs['domain_right_edge'] = ds.domain_right_edge.v
        h5f.attrs['domain_dimensions'] = ds.domain_dimensions
        h5f.attrs['field_names'] = [s.encode('utf-8') for s in field_names]
        
        # Add coordinate arrays
        x = np.linspace(ds.domain_left_edge[0].v, ds.domain_right_edge[0].v, nx)
        y = np.linspace(ds.domain_left_edge[1].v, ds.domain_right_edge[1].v, ny)
        z = np.linspace(ds.domain_left_edge[2].v, ds.domain_right_edge[2].v, nz)
        
        h5f.create_dataset('coordinates/x', data=x)
        h5f.create_dataset('coordinates/y', data=y)
        h5f.create_dataset('coordinates/z', data=z)
        
        # Add field data
        for field_name, data in field_data.items():
            h5f.create_dataset(f'fields/{field_name}', data=data, compression='gzip')
    
    # 3. Save flattened data as CSV (for small datasets or 2D slices)
    print(f"Saving flattened data as CSV: {output_prefix}_flattened.csv")
    
    # Create coordinate meshgrids
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten everything
    flat_data = {
        'x': X.flatten(),
        'y': Y.flatten(), 
        'z': Z.flatten()
    }
    
    for field_name, data in field_data.items():
        flat_data[field_name] = data.flatten()
    
    df = pd.DataFrame(flat_data)
    df.to_csv(f"{output_prefix}_flattened.csv", index=False)
    
    # 4. Save a 2D slice at z=middle for easier visualization/analysis
    z_mid = nz // 2
    print(f"Saving 2D slice at z={z_mid}: {output_prefix}_slice_z{z_mid}.csv")
    
    slice_data = {
        'x': X[:, :, z_mid].flatten(),
        'y': Y[:, :, z_mid].flatten()
    }
    
    for field_name, data in field_data.items():
        slice_data[field_name] = data[:, :, z_mid].flatten()
    
    df_slice = pd.DataFrame(slice_data)
    df_slice.to_csv(f"{output_prefix}_slice_z{z_mid}.csv", index=False)
    
    print(f"\nConversion complete! Files created:")
    print(f"  1. {output_prefix}.npz - NumPy format (best for ML)")
    print(f"  2. {output_prefix}.h5 - HDF5 format")
    print(f"  3. {output_prefix}_flattened.csv - Full 3D data as CSV")
    print(f"  4. {output_prefix}_slice_z{z_mid}.csv - 2D slice as CSV")
    
    # Show how to load the data back
    print(f"\nTo load in Python for ML:")
    print(f"  import numpy as np")
    print(f"  data = np.load('{output_prefix}.npz')")
    print(f"  # Access fields like: data['rho'], data['vel_x'], etc.")
    
    return field_data, field_names

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_amrex_to_ml.py <plotfile_directory>")
        print("Example: python convert_amrex_to_ml.py plt_hdf5_00100")
        sys.exit(1)
    
    plotfile_path = sys.argv[1]
    
    if not os.path.exists(plotfile_path):
        print(f"Error: {plotfile_path} does not exist")
        sys.exit(1)
    
    # Create output filename based on input
    output_prefix = f"ml_{os.path.basename(plotfile_path)}"
    
    try:
        convert_amrex_to_ml_formats(plotfile_path, output_prefix)
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)