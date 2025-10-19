#!/usr/bin/env python3
"""
Post-processing script to convert AMReX simulation outputs to ML-friendly formats

This script processes all 125 training cases and converts their plt* files to:
- NPZ format (compressed NumPy arrays)
- HDF5 format (hierarchical data)
- CSV format (for inspection)

The output will be organized for neural network training with input/output pairs.
"""

import os
import glob
import numpy as np
import h5py
import pandas as pd
import time
from pathlib import Path

def setup_ml_environment():
    """Setup the ML environment (marbles_ml conda env)"""
    try:
        import yt
        print(f" yt-project version: {yt.__version__}")
        return True
    except ImportError:
        print(" yt-project not found. Please run:")
        print("  conda activate marbles_ml")
        return False

def extract_parameters_from_case_name(case_name):
    """
    Extract parameter indices from case directory name
    e.g., case_062_nu_2_temp_2_geom_3 -> (62, 2, 2, 3)
    """
    parts = case_name.split('_')
    case_id = int(parts[1])
    nu_idx = int(parts[3])
    temp_idx = int(parts[5])
    geom_id = int(parts[7])
    return case_id, nu_idx, temp_idx, geom_id

def get_parameter_values(nu_idx, temp_idx):
    """
    Get actual parameter values from indices
    """
    # Reference values from isothermal_cracks.inp
    nu_ref = 4.857e-3
    temp_ref = 0.03333
    
    # Generate the same sweep ranges as in training data generation
    nu_values = []
    temp_values = []
    
    # Generate 5 viscosity values
    nu_min, nu_max = nu_ref/1.5, nu_ref*1.5
    for i in range(5):
        nu = nu_min + i * (nu_max - nu_min) / 4
        nu_values.append(nu)
    
    # Generate 5 temperature values  
    temp_min, temp_max = temp_ref/1.5, temp_ref*1.5
    for i in range(5):
        temp = temp_min + i * (temp_max - temp_min) / 4
        temp_values.append(temp)
    
    return nu_values[nu_idx], temp_values[temp_idx]

def convert_amrex_to_ml_format(plt_file, output_base_name):
    """
    Convert AMReX plotfile to multiple ML formats
    
    Parameters:
    -----------
    plt_file : str
        Path to AMReX plotfile (plt00000, etc.)
    output_base_name : str
        Base name for output files (without extension)
        
    Returns:
    --------
    dict : Information about converted files
    """
    try:
        import yt
        
        # Load AMReX data with yt
        ds = yt.load(plt_file)
        
        # Get the data as a uniform grid
        cg = ds.covering_grid(level=0, left_edge=ds.domain_left_edge, 
                             dims=ds.domain_dimensions)
        
        # Extract all field data
        field_data = {}
        field_info = {}
        
        # Get all available fields
        available_fields = [field for field in ds.field_list if field[0] != 'index']
        
        print(f"    Fields available: {len(available_fields)}")
        
        for field in available_fields:
            field_name = field[1]  # Get just the field name, not the field type
            try:
                data = np.array(cg[field])
                field_data[field_name] = data
                field_info[field_name] = {
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'mean': float(np.mean(data))
                }
            except Exception as e:
                print(f"      Warning: Could not extract field {field_name}: {e}")
        
        # Get grid information
        grid_info = {
            'domain_dimensions': ds.domain_dimensions.tolist(),
            'domain_left_edge': ds.domain_left_edge.tolist(),
            'domain_right_edge': ds.domain_right_edge.tolist(),
            'time': float(ds.current_time) if hasattr(ds, 'current_time') else 0.0,
            'cycle': int(ds.parameters.get('amr.plot_int', 0)) if hasattr(ds, 'parameters') else 0
        }
        
        # 1. Save as NPZ (compressed NumPy format)
        npz_file = f"{output_base_name}.npz"
        np.savez_compressed(npz_file, 
                           grid_info=grid_info,
                           field_info=field_info,
                           **field_data)
        
        # 2. Save as HDF5
        h5_file = f"{output_base_name}.h5"
        with h5py.File(h5_file, 'w') as f:
            # Store grid information
            grid_grp = f.create_group('grid_info')
            for key, value in grid_info.items():
                grid_grp.attrs[key] = value
            
            # Store field data
            fields_grp = f.create_group('fields')
            for field_name, data in field_data.items():
                fields_grp.create_dataset(field_name, data=data, compression='gzip')
                
                # Store field metadata
                field_grp = fields_grp[field_name]
                for key, value in field_info[field_name].items():
                    field_grp.attrs[key] = value
        
        # 3. Save summary as CSV (for inspection)
        csv_file = f"{output_base_name}_summary.csv"
        summary_data = []
        for field_name, info in field_info.items():
            summary_data.append({
                'field': field_name,
                'shape': str(info['shape']),
                'dtype': info['dtype'],
                'min': info['min'],
                'max': info['max'],
                'mean': info['mean']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(csv_file, index=False)
        
        # Get file sizes
        file_info = {
            'npz_size': os.path.getsize(npz_file),
            'h5_size': os.path.getsize(h5_file),
            'csv_size': os.path.getsize(csv_file),
            'num_fields': len(field_data),
            'grid_size': grid_info['domain_dimensions']
        }
        
        return file_info
        
    except Exception as e:
        print(f"     Error converting {plt_file}: {e}")
        return None

def process_training_case(case_dir, output_dir):
    """
    Process a single training case directory
    
    Parameters:
    -----------
    case_dir : str
        Path to case directory (e.g., training_data/case_000_nu_0_temp_0_geom_1)
    output_dir : str
        Output directory for processed data
        
    Returns:
    --------
    dict : Processing results
    """
    case_name = os.path.basename(case_dir)
    print(f"Processing {case_name}...")
    
    # Extract parameters from case name
    case_id, nu_idx, temp_idx, geom_id = extract_parameters_from_case_name(case_name)
    nu_val, temp_val = get_parameter_values(nu_idx, temp_idx)
    alpha_val = nu_val / 0.7
    
    # Find all plt files in the case directory
    plt_pattern = os.path.join(case_dir, "plt*")
    plt_files = sorted([f for f in glob.glob(plt_pattern) if os.path.isdir(f)])
    
    if not plt_files:
        print(f"    No plt files found in {case_dir}")
        return {'case_id': case_id, 'status': 'no_data', 'files_processed': 0}
    
    print(f"  Found {len(plt_files)} plt files")
    
    # Create output directory for this case
    case_output_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)
    
    # Process each plt file
    processed_files = 0
    file_infos = []
    
    for i, plt_file in enumerate(plt_files):
        plt_name = os.path.basename(plt_file)
        
        # Create output base name
        output_base = os.path.join(case_output_dir, f"{plt_name}_ml")
        
        print(f"    [{i+1:3d}/{len(plt_files)}] Converting {plt_name}...")
        
        file_info = convert_amrex_to_ml_format(plt_file, output_base)
        
        if file_info:
            # Add metadata
            file_info.update({
                'case_id': case_id,
                'nu_index': nu_idx,
                'temp_index': temp_idx,  
                'geom_id': geom_id,
                'nu_value': nu_val,
                'temp_value': temp_val,
                'alpha_value': alpha_val,
                'plt_file': plt_name,
                'timestep': i
            })
            file_infos.append(file_info)
            processed_files += 1
    
    # Save metadata for this case
    metadata_file = os.path.join(case_output_dir, f"{case_name}_metadata.json")
    import json
    with open(metadata_file, 'w') as f:
        json.dump({
            'case_info': {
                'case_id': case_id,
                'case_name': case_name,
                'nu_index': nu_idx,
                'temp_index': temp_idx,
                'geom_id': geom_id,
                'nu_value': nu_val,
                'temp_value': temp_val,
                'alpha_value': alpha_val
            },
            'files_processed': processed_files,
            'total_plt_files': len(plt_files),
            'file_details': file_infos
        }, f, indent=2)
    
    return {
        'case_id': case_id,
        'status': 'success',
        'files_processed': processed_files,
        'total_files': len(plt_files)
    }

def process_all_training_data(training_data_dir="training_data", output_dir="ml_training_data"):
    """
    Process all training data cases
    """
    print("AMReX to ML Format Conversion")
    print("=" * 60)
    
    # Check environment
    if not setup_ml_environment():
        return
    
    # Find all case directories
    case_pattern = os.path.join(training_data_dir, "case_*")
    case_dirs = sorted(glob.glob(case_pattern))
    
    if not case_dirs:
        print(f" No case directories found in {training_data_dir}")
        return
    
    print(f"Found {len(case_dirs)} case directories to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each case
    results = []
    start_time = time.time()
    
    for i, case_dir in enumerate(case_dirs):
        try:
            result = process_training_case(case_dir, output_dir)
            results.append(result)
            
            # Progress update every 25 cases
            if (i + 1) % 25 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(case_dirs) - i - 1)
                successful = sum(1 for r in results if r['status'] == 'success')
                print(f"   Progress: {i+1}/{len(case_dirs)} | "
                      f"Successful: {successful} | "
                      f"ETA: {remaining/60:.1f} min")
                
        except Exception as e:
            print(f"   Error processing {os.path.basename(case_dir)}: {e}")
            results.append({
                'case_id': i,
                'status': 'error',
                'files_processed': 0,
                'error': str(e)
            })
    
    # Final summary
    total_time = time.time() - start_time
    successful_cases = sum(1 for r in results if r['status'] == 'success')
    total_files = sum(r.get('files_processed', 0) for r in results)
    
    print(f"\n" + "=" * 60)
    print(f"Processing Complete!")
    print(f"  Total cases: {len(case_dirs)}")
    print(f"  Successful cases: {successful_cases}")
    print(f"  Total files converted: {total_files}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Output directory: {output_dir}")
    
    # Save overall summary
    summary_file = os.path.join(output_dir, "processing_summary.json")
    import json
    with open(summary_file, 'w') as f:
        json.dump({
            'processing_info': {
                'total_cases': len(case_dirs),
                'successful_cases': successful_cases,
                'total_files_converted': total_files,
                'processing_time_minutes': total_time/60,
                'output_directory': output_dir
            },
            'case_results': results
        }, f, indent=2)
    
    print(f"  Summary saved: {summary_file}")

if __name__ == "__main__":
    print(" Starting AMReX to ML format conversion...")
    process_all_training_data()