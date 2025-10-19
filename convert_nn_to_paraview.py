#!/usr/bin/env python3
"""
Convert Neural Network            if len(field_data.shape) == 4 and field_data.shape[3] == 3:  # Vector field (nx, ny, nz, 3)
                f.write(f"VECTORS {field_name} float\n")
                
                # Write vector data in VTK order (k, j, i)
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            vx = field_data[i, j, k, 0]
                            vy = field_data[i, j, k, 1]
                            vz = field_data[i, j, k, 2]
                            f.write(f"{vx} {vy} {vz}\n")ons to ParaView-compatible VTK format

This script converts the .npz neural network predictions to VTK files
that can be opened and visualized in ParaView.
"""

import numpy as np
import os
from pathlib import Path

def write_vtk_structured_grid(filename, data_dict, dimensions, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Write a VTK structured grid file
    
    Parameters:
    -----------
    filename : str
        Output VTK filename
    data_dict : dict
        Dictionary of field data {field_name: numpy_array}
    dimensions : tuple
        Grid dimensions (nx, ny, nz)
    spacing : tuple
        Grid spacing
    origin : tuple
        Grid origin
    """
    nx, ny, nz = dimensions
    
    with open(filename, 'w') as f:
        # VTK header
        f.write("# vtk DataFile Version 3.0\\n")
        f.write("Neural Network LBM Predictions\\n")
        f.write("ASCII\\n")
        f.write("DATASET STRUCTURED_POINTS\\n")
        
        # Grid definition
        f.write(f"DIMENSIONS {nx} {ny} {nz}\\n")
        f.write(f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\\n")
        f.write(f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\\n")
        
        # Point data
        f.write(f"POINT_DATA {nx * ny * nz}\\n")
        
        # Write each field
        for field_name, field_data in data_dict.items():
            
            if len(field_data.shape) == 4 and field_data.shape[3] == 3:  # Vector field (nx, ny, nz, 3)
                f.write(f"VECTORS {field_name} float\\n")
                
                # Write vector data in VTK order (k, j, i)
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            vx = field_data[i, j, k, 0]
                            vy = field_data[i, j, k, 1]
                            vz = field_data[i, j, k, 2]
                            f.write(f"{vx} {vy} {vz}\\n")
                            
            elif len(field_data.shape) == 4 and field_data.shape[3] == 1:  # Scalar field (nx, ny, nz, 1)
                f.write(f"SCALARS {field_name} float\\n")
                f.write("LOOKUP_TABLE default\\n")
                
                # Write scalar data in VTK order (k, j, i)
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            value = field_data[i, j, k, 0]
                            f.write(f"{value}\\n")
                            
            elif len(field_data.shape) == 3:  # Scalar field (nx, ny, nz)
                f.write(f"SCALARS {field_name} float\\n")
                f.write("LOOKUP_TABLE default\\n")
                
                # Write scalar data in VTK order (k, j, i)
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            value = field_data[i, j, k]
                            f.write(f"{value}\\n")

def convert_npz_to_vtk(npz_file, output_dir):
    """
    Convert a single .npz file to VTK format (one file per timestep)
    """
    print(f" Converting {npz_file}")
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    
    case_name = Path(npz_file).stem.replace('_nn_predictions', '')
    timesteps = data['timesteps']
    
    # Create output directory for this case
    case_output_dir = os.path.join(output_dir, f"{case_name}_vtk")
    os.makedirs(case_output_dir, exist_ok=True)
    
    print(f"    Output directory: {case_output_dir}")
    print(f"    Converting {len(timesteps)} timesteps...")
    
    # Grid dimensions
    dimensions = (60, 40, 30)
    
    # Convert each timestep
    for t_idx, timestep in enumerate(timesteps):
        
        # Prepare field data for this timestep
        fields = {}
        
        # Velocity (vector field)
        if 'velocity_fields' in data:
            vel_data = data['velocity_fields'][t_idx]  # Shape: (60, 40, 30, 3)
            fields['velocity'] = vel_data
        
        # Heat flux (vector field)
        if 'heat_flux_fields' in data:
            hf_data = data['heat_flux_fields'][t_idx]   # Shape: (60, 40, 30, 3)
            fields['heat_flux'] = hf_data
        
        # Scalar fields
        for field_name in ['density', 'energy', 'temperature']:
            data_key = f"{field_name}_fields"
            if data_key in data:
                field_data = data[data_key][t_idx]  # Shape: (60, 40, 30, 1)
                fields[field_name] = field_data
        
        # Write VTK file
        vtk_filename = os.path.join(case_output_dir, f"{case_name}_t{t_idx:05d}.vtk")
        write_vtk_structured_grid(vtk_filename, fields, dimensions)
        
        if t_idx % 20 == 0:  # Progress update
            print(f"    Converted timestep {t_idx}/{len(timesteps)}")
    
    print(f"    Completed conversion: {len(timesteps)} VTK files created")
    
    # Create ParaView collection file for easy loading
    create_paraview_collection(case_output_dir, case_name, len(timesteps))

def create_paraview_collection(output_dir, case_name, num_timesteps):
    """
    Create a ParaView collection (.pvd) file for easy time series loading
    """
    collection_file = os.path.join(output_dir, f"{case_name}_time_series.pvd")
    
    with open(collection_file, 'w') as f:
        f.write('<?xml version="1.0"?>\\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\\n')
        f.write('  <Collection>\\n')
        
        for t_idx in range(num_timesteps):
            vtk_file = f"{case_name}_t{t_idx:05d}.vtk"
            f.write(f'    <DataSet timestep="{t_idx}" group="" part="0" file="{vtk_file}"/>\\n')
        
        f.write('  </Collection>\\n')
        f.write('</VTKFile>\\n')
    
    print(f"    Created ParaView collection: {collection_file}")

def main():
    """
    Main conversion function
    """
    print(" Neural Network Predictions → ParaView VTK Converter")
    print("=" * 60)
    
    # Input and output directories
    input_dir = "validation/neural_network_predictions"
    output_dir = "validation/paraview_vtk"
    
    if not os.path.exists(input_dir):
        print(" No neural network predictions found!")
        print(f"   Expected directory: {input_dir}")
        return
    
    # Find all .npz files
    npz_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npz')]
    
    if not npz_files:
        print(f" No .npz files found in {input_dir}")
        return
    
    print(f" Found {len(npz_files)} prediction files to convert")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each file
    for npz_file in npz_files:
        try:
            convert_npz_to_vtk(npz_file, output_dir)
        except Exception as e:
            print(f" Error converting {npz_file}: {e}")
            continue
    
    print(f"\\n Conversion completed!")
    print(f" ParaView VTK files saved to: {output_dir}")
    print(f"\\n To view in ParaView:")
    print(f"   1. Open ParaView")
    print(f"   2. Open the .pvd files in {output_dir}")
    print(f"   3. Each .pvd file contains the full time series for one validation case")
    print(f"   4. Use the time controls to animate through timesteps")

if __name__ == "__main__":
    main()