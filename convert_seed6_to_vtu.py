#!/usr/bin/env python3
"""
Convert Validation Seed6 Neural Network Predictions to ParaView VTU format

This script processes the validation_seed6 results to create
ParaView-compatible VTU files for visualization of the unseen geometry validation.
(Copy of convert_nn_to_vtu.py with only directory paths changed)
"""

import numpy as np
import os
from pathlib import Path

def write_vtu_structured_grid(filename, data_dict, dimensions, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Write a VTU (VTK Unstructured Grid) file in XML format
    
    Parameters:
    -----------
    filename : str
        Output VTU filename
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
    num_points = nx * ny * nz
    
    with open(filename, 'w') as f:
        # VTU XML header
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{num_points}" NumberOfCells="{(nx-1)*(ny-1)*(nz-1)}">\n')
        
        # Points section
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        
        # Write point coordinates
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = origin[0] + i * spacing[0]
                    y = origin[1] + j * spacing[1]
                    z = origin[2] + k * spacing[2]
                    f.write(f'          {x} {y} {z}\n')
        
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')
        
        # Cells section (hexahedral cells)
        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        
        # Write cell connectivity (hexahedral)
        for k in range(nz-1):
            for j in range(ny-1):
                for i in range(nx-1):
                    # Hexahedral cell vertices (VTK ordering)
                    v0 = k*nx*ny + j*nx + i
                    v1 = k*nx*ny + j*nx + (i+1)
                    v2 = k*nx*ny + (j+1)*nx + (i+1)
                    v3 = k*nx*ny + (j+1)*nx + i
                    v4 = (k+1)*nx*ny + j*nx + i
                    v5 = (k+1)*nx*ny + j*nx + (i+1)
                    v6 = (k+1)*nx*ny + (j+1)*nx + (i+1)
                    v7 = (k+1)*nx*ny + (j+1)*nx + i
                    f.write(f'          {v0} {v1} {v2} {v3} {v4} {v5} {v6} {v7}\n')
        
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        
        # Write cell offsets
        for cell_id in range((nx-1)*(ny-1)*(nz-1)):
            f.write(f'          {(cell_id+1)*8}\n')
        
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        
        # Write cell types (12 = VTK_HEXAHEDRON)
        for cell_id in range((nx-1)*(ny-1)*(nz-1)):
            f.write('          12\n')
        
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')
        
        # Point data section
        f.write('      <PointData>\n')
        
        # Write each field
        for field_name, field_data in data_dict.items():
            
            if len(field_data.shape) == 4 and field_data.shape[3] == 3:  # Vector field
                f.write(f'        <DataArray type="Float32" Name="{field_name}" NumberOfComponents="3" format="ascii">\n')
                
                # Write vector data in VTK point order
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            vx = field_data[i, j, k, 0]
                            vy = field_data[i, j, k, 1]
                            vz = field_data[i, j, k, 2]
                            f.write(f'          {vx} {vy} {vz}\n')
                            
            elif len(field_data.shape) == 4 and field_data.shape[3] == 1:  # Scalar field
                f.write(f'        <DataArray type="Float32" Name="{field_name}" format="ascii">\n')
                
                # Write scalar data in VTK point order
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            value = field_data[i, j, k, 0]
                            f.write(f'          {value}\n')
            
            elif len(field_data.shape) == 3:  # Scalar field without extra dimension
                f.write(f'        <DataArray type="Float32" Name="{field_name}" format="ascii">\n')
                
                # Write scalar data in VTK point order
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            value = field_data[i, j, k]
                            f.write(f'          {value}\n')
            
            f.write('        </DataArray>\n')
        
        f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')

def convert_npz_to_vtu(npz_file, output_dir):
    """
    Convert a single .npz file to VTU format (one file per timestep)
    """
    print(f"Converting {npz_file}")
    
    # Load data
    data = np.load(npz_file, allow_pickle=True)
    
    case_name = Path(npz_file).stem.replace('_nn_predictions', '')
    timesteps = data['timesteps']
    
    # Create output directory for this case
    case_output_dir = os.path.join(output_dir, f"{case_name}_vtu")
    os.makedirs(case_output_dir, exist_ok=True)
    
    print(f"   Output directory: {case_output_dir}")
    print(f"   Converting {len(timesteps)} timesteps...")
    
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
        
        # Write VTU file
        vtu_filename = os.path.join(case_output_dir, f"{case_name}_t{t_idx:05d}.vtu")
        write_vtu_structured_grid(vtu_filename, fields, dimensions)
        
        if t_idx % 20 == 0:  # Progress update
            print(f"   Converted timestep {t_idx}/{len(timesteps)}")
    
    print(f"   Completed conversion: {len(timesteps)} VTU files created")
    
    # Create ParaView collection file for easy loading
    create_paraview_collection_vtu(case_output_dir, case_name, len(timesteps))

def create_paraview_collection_vtu(output_dir, case_name, num_timesteps):
    """
    Create a ParaView collection (.pvd) file for VTU time series
    """
    collection_file = os.path.join(output_dir, f"{case_name}_time_series.pvd")
    
    with open(collection_file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write('  <Collection>\n')
        
        for t_idx in range(num_timesteps):
            vtu_file = f"{case_name}_t{t_idx:05d}.vtu"
            f.write(f'    <DataSet timestep="{t_idx}" group="" part="0" file="{vtu_file}"/>\n')
        
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')
    
    print(f"   Created ParaView collection: {collection_file}")

def main():
    """
    Main conversion function
    """
    print("Seed6 Validation Results to ParaView VTU Converter")
    print("=" * 60)
    
    # Input and output directories (ONLY CHANGE: seed6 paths)
    input_dir = "validation_seed6/neural_network_predictions"
    output_dir = "validation_seed6/paraview_vtu"
    
    if not os.path.exists(input_dir):
        print("No seed6 neural network predictions found!")
        print(f"   Expected directory: {input_dir}")
        return
    
    # Find all .npz files
    npz_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npz')]
    
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} prediction files to convert")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each file
    for npz_file in npz_files:
        try:
            convert_npz_to_vtu(npz_file, output_dir)
        except Exception as e:
            print(f"Error converting {npz_file}: {e}")
            continue
    
    print(f"\nConversion completed!")
    print(f"ParaView VTU files saved to: {output_dir}")
    print(f"\nTo view in ParaView:")
    print(f"   1. Open ParaView")
    print(f"   2. Open the .pvd files in {output_dir}")
    print(f"   3. VTU format will load automatically without reader selection")
    print(f"   4. Each .pvd file contains the full time series for one seed6 validation case")
    print(f"   5. Use the time controls to animate through timesteps")

if __name__ == "__main__":
    main()