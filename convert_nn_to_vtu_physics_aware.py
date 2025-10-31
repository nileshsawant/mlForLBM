#!/usr/bin/env python3
"""
Convert Physics-Aware Neural Network Predictions to ParaView-compatible VTU format

This script converts the .npz physics-aware neural network predictions to VTU files
for easy inspection in ParaView.

It looks for outputs in the physics-aware validation directories and a custom predictions
folder created by `predict_and_visualize_physics_aware.py`.

Author: automated copy adapted from convert_nn_to_vtu_enhanced.py
"""

import numpy as np
import os
from pathlib import Path


def write_vtu_structured_grid(filename, data_dict, dimensions, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    nx, ny, nz = dimensions
    num_points = nx * ny * nz

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{num_points}" NumberOfCells="{(nx-1)*(ny-1)*(nz-1)}">\n')

        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    x = origin[0] + i * spacing[0]
                    y = origin[1] + j * spacing[1]
                    z = origin[2] + k * spacing[2]
                    f.write(f'          {x} {y} {z}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        # Cells
        f.write('      <Cells>\n')
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for k in range(nz-1):
            for j in range(ny-1):
                for i in range(nx-1):
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
        for cell_id in range((nx-1)*(ny-1)*(nz-1)):
            f.write(f'          {(cell_id+1)*8}\n')
        f.write('        </DataArray>\n')
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for cell_id in range((nx-1)*(ny-1)*(nz-1)):
            f.write('          12\n')
        f.write('        </DataArray>\n')
        f.write('      </Cells>\n')

        # Point data
        f.write('      <PointData>\n')
        for field_name, field_data in data_dict.items():
            if len(field_data.shape) == 4 and field_data.shape[3] == 3:
                f.write(f'        <DataArray type="Float32" Name="{field_name}" NumberOfComponents="3" format="ascii">\n')
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            vx = field_data[i, j, k, 0]
                            vy = field_data[i, j, k, 1]
                            vz = field_data[i, j, k, 2]
                            f.write(f'          {vx} {vy} {vz}\n')
            elif len(field_data.shape) == 4 and field_data.shape[3] == 1:
                f.write(f'        <DataArray type="Float32" Name="{field_name}" format="ascii">\n')
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            value = field_data[i, j, k, 0]
                            f.write(f'          {value}\n')
            elif len(field_data.shape) == 3:
                f.write(f'        <DataArray type="Float32" Name="{field_name}" format="ascii">\n')
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
    print(f" Converting {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    case_name = Path(npz_file).stem.replace('_nn_predictions', '')
    case_output_dir = os.path.join(output_dir, f"{case_name}_vtu")
    os.makedirs(case_output_dir, exist_ok=True)
    timesteps = data['timesteps']
    dimensions = (60, 40, 30)

    for t_idx, _ in enumerate(timesteps):
        fields = {}
        if 'velocity_fields' in data:
            fields['velocity'] = data['velocity_fields'][t_idx]
        if 'heat_flux_fields' in data:
            fields['heat_flux'] = data['heat_flux_fields'][t_idx]
        for field_name in ['density', 'energy', 'temperature']:
            key = f"{field_name}_fields"
            if key in data:
                fields[field_name] = data[key][t_idx]

        vtu_filename = os.path.join(case_output_dir, f"{case_name}_t{t_idx:05d}.vtu")
        write_vtu_structured_grid(vtu_filename, fields, dimensions)
        if t_idx % 20 == 0:
            print(f"    Converted timestep {t_idx}/{len(timesteps)}")

    create_paraview_collection_vtu(case_output_dir, case_name, len(timesteps))
    print(f"    Completed conversion: {len(timesteps)} VTU files created")


def create_paraview_collection_vtu(output_dir, case_name, num_timesteps):
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
    print(f"    Created ParaView collection: {collection_file}")


def process_validation_directory(validation_type, input_dir, output_dir):
    print(f"\n Processing {validation_type} validation results...")
    print(f"    Input: {input_dir}")
    print(f"    Output: {output_dir}")
    if not os.path.exists(input_dir):
        print(f"     Directory not found: {input_dir}")
        return False
    npz_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.npz')]
    if not npz_files:
        print(f"     No .npz files found in {input_dir}")
        return False
    print(f"    Found {len(npz_files)} prediction files to convert")
    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    for npz_file in npz_files:
        try:
            convert_npz_to_vtu(npz_file, output_dir)
            success_count += 1
        except Exception as e:
            print(f"    Error converting {npz_file}: {e}")
            continue
    print(f"    Successfully converted {success_count}/{len(npz_files)} files")
    return success_count > 0


def main():
    print(" Physics-Aware Neural Network Predictions to ParaView VTU Converter")
    print("=" * 70)

    conversion_tasks = [
        {
            "type": "PHYSICS-AWARE (Seen Geometries)",
            "input_dir": "validation_physics_aware/neural_network_predictions",
            "output_dir": "validation_physics_aware/paraview_vtu"
        },
        {
            "type": "PHYSICS-AWARE GENERALIZATION (Unseen Geometry)",
            "input_dir": "validation_seed6_physics_aware/neural_network_predictions",
            "output_dir": "validation_seed6_physics_aware/paraview_vtu"
        },
        {
            "type": "CUSTOM PREDICTIONS",
            "input_dir": "custom_predictions_physics_aware",
            "output_dir": "custom_predictions_physics_aware/paraview_vtu"
        }
    ]

    successful_conversions = 0
    for task in conversion_tasks:
        success = process_validation_directory(task["type"], task["input_dir"], task["output_dir"])
        if success:
            successful_conversions += 1

    print(f"\n Physics-Aware Conversion Summary")
    print("=" * 70)
    if successful_conversions == 0:
        print(" No physics-aware prediction results found to convert!")
    else:
        print(f" Successfully processed {successful_conversions}/{len(conversion_tasks)} prediction sources")
        print("\n ParaView Visualization Guide:")
        if successful_conversions >= 1:
            print("   • For seen-geometry validation: open .pvd files in validation_physics_aware/paraview_vtu/")
        if successful_conversions >= 2:
            print("   • For unseen-geometry validation: open .pvd files in validation_seed6_physics_aware/paraview_vtu/")
        if successful_conversions >= 3:
            print("   • For custom predictions: open .pvd files in custom_predictions_physics_aware/paraview_vtu/")


if __name__ == '__main__':
    main()
