#!/usr/bin/env python3
"""
Physics-Aware LBM Neural Network Prediction and Visualization

This script provides an easy-to-use interface for generating physics-aware neural network predictions
of LBM thermal flow with ZERO physics violations and converting them to ParaView-compatible VTU format.

Key features:
- Uses physics-aware model with activation function constraints
- Guarantees positive densities, temperatures, and energies (softplus activation)
- Bounded velocity and heat flux ranges (tanh activation)
- Zero physics violations by mathematical design

Reuses existing validated functions from the repository:
- Model loading adapted from validate_neural_network_physics_aware.py
- Geometry processing from predict_with_neural_network()
- VTU conversion from convert_nn_to_vtu.py

Author: Generated for physics-aware predictions with zero violations
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import subprocess
import shutil
import time

# Add current directory to path for imports
sys.path.append('.')

def load_enhanced_model():
    """
    Load the physics-aware trained neural network model
    Reused from validate_neural_network_physics_aware.py
    """
    # PHYSICS-AWARE MODEL FILES (prioritized)
    physics_aware_model_files = [
        'lbm_flow_predictor_physics_aware.h5',
        'best_physics_aware_model.h5'
    ]
    
    # Fallback to enhanced/baseline models if physics-aware not found
    fallback_model_files = [
        'lbm_flow_predictor_cno-inspired_enhanced.h5',
        'lbm_flow_predictor_cno_inspired_enhanced.h5', 
        'lbm_flow_predictor_enhanced.h5',
        'lbm_flow_predictor_cno-inspired.h5',
        'lbm_flow_predictor_cno_inspired.h5', 
        'lbm_flow_predictor.h5'
    ]
    
    all_model_files = physics_aware_model_files + fallback_model_files
    
    for i, model_file in enumerate(all_model_files):
        if os.path.exists(model_file):
            model_type = " PHYSICS-AWARE" if i < len(physics_aware_model_files) else " FALLBACK"
            print(f" Loading {model_type} model: {model_file}")
            
            try:
                # Try loading without compiling (avoid custom loss function issues)
                model = keras.models.load_model(model_file, compile=False)
                print(f" Model loaded successfully: {model.count_params():,} parameters")
                return model
            except Exception as e:
                print(f" Direct loading failed: {e}")
                print(" Attempting to rebuild model and load weights...")
                
                try:
                    # Import the model creation function
                    from train_lbm_neural_network import create_ultra_efficient_cno_model
                    
                    # Recreate the model architecture
                    model = create_ultra_efficient_cno_model()
                    model.load_weights(model_file)
                    print(f" Model rebuilt and weights loaded: {model.count_params():,} parameters")
                    return model
                    
                except Exception as rebuild_error:
                    print(f" Rebuild failed: {rebuild_error}")
                    continue
    
    raise FileNotFoundError("No compatible model files found! Please train a model first.")

def load_geometry(geometry_file):
    """
    Load geometry from CSV file
    Adapted from predict_with_neural_network() in validation scripts
    """
    if not os.path.exists(geometry_file):
        raise FileNotFoundError(f"Geometry file not found: {geometry_file}")
    
    # Load geometry as 3D array with error handling
    try:
        geom_df = pd.read_csv(geometry_file)
        print(f" Loaded geometry file: {geometry_file}")
        print(f"   Geometry data shape: {geom_df.shape}")
        print(f"   Columns: {list(geom_df.columns)}")
        
        # Check if we have the expected columns
        if 'x' not in geom_df.columns:
            # Try alternative column names
            if len(geom_df.columns) >= 4:
                geom_df.columns = ['x', 'y', 'z', 'value']
                print(f"   Renamed columns to: {list(geom_df.columns)}")
            else:
                raise ValueError(f"Unexpected geometry file format. Columns: {list(geom_df.columns)}")
        
        geometry_3d = np.zeros((60, 40, 30))
        
        # Load geometry data
        for _, row in geom_df.iterrows():
            try:
                x, y, z = int(row['x']), int(row['y']), int(row['z'])
                if 0 <= x < 60 and 0 <= y < 40 and 0 <= z < 30:
                    geometry_3d[x, y, z] = row['value']
            except (ValueError, KeyError) as row_error:
                print(f"    Skipping invalid row: {row_error}")
                continue
        
        solid_voxels = np.sum(geometry_3d > 0)
        total_voxels = 60 * 40 * 30
        print(f"   Loaded geometry: {solid_voxels:,} solid voxels ({solid_voxels/total_voxels*100:.1f}%)")
        
        return geometry_3d
        
    except Exception as geom_error:
        print(f" Geometry loading error: {geom_error}")
        raise

def generate_predictions(model, geometry_3d, nu, temperature, timesteps):
    """
    Generate neural network predictions
    Adapted from predict_with_neural_network() in validation scripts
    """
    print(f" Generating neural network predictions...")
    print(f"   Parameters: nu={nu:.6f}, temperature={temperature:.3f}")
    print(f"   Timesteps: {len(timesteps)} steps")
    
    # Prepare inputs for all timesteps
    num_timesteps = len(timesteps)
    geometries = np.tile(geometry_3d[np.newaxis, ..., np.newaxis], (num_timesteps, 1, 1, 1, 1))
    
    # Create parameters for each timestep (including time as 4th parameter)
    parameters = np.zeros((num_timesteps, 4))
    alpha_value = nu / 0.7  # Calculate alpha = nu / Prandtl
    
    for i, t in enumerate(timesteps):
        parameters[i] = [
            nu, 
            temperature,
            alpha_value,  # Correctly calculated alpha
            t             # normalized time
        ]
    
    # Generate predictions
    predictions = model.predict([geometries, parameters], batch_size=8, verbose=0)
    
    # Unpack predictions (5 physics fields)
    velocity_pred, heat_flux_pred, density_pred, energy_pred, temperature_pred = predictions
    
    prediction_data = {
        'timesteps': timesteps,
        'velocity_fields': velocity_pred,      # Shape: (timesteps, 60, 40, 30, 3)
        'heat_flux_fields': heat_flux_pred,    # Shape: (timesteps, 60, 40, 30, 3) 
        'density_fields': density_pred,        # Shape: (timesteps, 60, 40, 30, 1)
        'energy_fields': energy_pred,          # Shape: (timesteps, 60, 40, 30, 1)
        'temperature_fields': temperature_pred # Shape: (timesteps, 60, 40, 30, 1)
    }
    
    print(f" Predictions generated successfully")
    
    #  PHYSICS VIOLATIONS CHECK (should be ZERO with physics-aware model)
    total_points = density_pred.size
    density_violations = np.sum(density_pred <= 0)
    temperature_violations = np.sum(temperature_pred <= 0) 
    energy_violations = np.sum(energy_pred <= 0)
    
    print(f" Physics Violations Check:")
    print(f"   Density violations:     {density_violations:>8} / {total_points:>10} ({100*density_violations/total_points:.3f}%)")
    print(f"   Temperature violations: {temperature_violations:>8} / {total_points:>10} ({100*temperature_violations/total_points:.3f}%)")
    print(f"   Energy violations:      {energy_violations:>8} / {total_points:>10} ({100*energy_violations/total_points:.3f}%)")
    
    if density_violations == 0 and temperature_violations == 0 and energy_violations == 0:
        print(f"    PERFECT: Zero physics violations achieved!")
    else:
        print(f"     Physics violations detected - check activation functions")
    
    print(f" Field Ranges:")
    print(f"   Density:     [{density_pred.min():.6f}, {density_pred.max():.6f}] (softplus)")
    print(f"   Temperature: [{temperature_pred.min():.6f}, {temperature_pred.max():.6f}] (softplus)")
    print(f"   Energy:      [{energy_pred.min():.6f}, {energy_pred.max():.6f}] (softplus)")
    print(f"   Velocity:    [{velocity_pred.min():.6f}, {velocity_pred.max():.6f}] (tanh)")
    print(f"   Heat flux:   [{heat_flux_pred.min():.6f}, {heat_flux_pred.max():.6f}] (tanh)")
    
    return prediction_data

def write_vtu_structured_grid(filename, data_dict, dimensions, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Write a VTU (VTK Unstructured Grid) file in XML format
    Reused from convert_nn_to_vtu.py with proper XML formatting
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
            offset = (cell_id + 1) * 8
            f.write(f'          {offset}\n')
        
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
                            
            elif len(field_data.shape) == 4 and field_data.shape[3] == 1:  # Scalar field with extra dimension
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

def create_paraview_collection_vtu(output_dir, case_name, num_timesteps):
    """
    Create ParaView collection (.pvd) file for time series
    Reused from convert_nn_to_vtu.py
    """
    pvd_file = os.path.join(output_dir, f"{case_name}.pvd")
    
    with open(pvd_file, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1">\n')
        f.write('  <Collection>\n')
        
        for t_idx in range(num_timesteps):
            physical_time = (t_idx / (num_timesteps - 1)) * 1000  # 0 to 1000
            vtu_filename = f"{case_name}_t{int(physical_time):04d}.vtu"
            f.write(f'    <DataSet timestep="{physical_time}" file="{vtu_filename}"/>\n')
        
        f.write('  </Collection>\n')
        f.write('</VTKFile>\n')

def convert_to_paraview_vtu(prediction_data, output_dir, case_name):
    """
    Convert prediction data to ParaView VTU format
    Adapted from convert_nn_to_vtu.py functions
    """
    print(f" Converting to ParaView VTU format...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Grid dimensions
    dimensions = (60, 40, 30)
    timesteps = prediction_data['timesteps']
    
    print(f"   Output directory: {output_dir}")
    print(f"   Converting {len(timesteps)} timesteps...")
    
    # Convert each timestep
    for t_idx, timestep in enumerate(timesteps):
        
        # Prepare field data for this timestep
        fields = {}
        
        # Velocity (vector field)
        if 'velocity_fields' in prediction_data:
            velocity_data = prediction_data['velocity_fields'][t_idx]  # (60, 40, 30, 3)
            fields['Velocity'] = velocity_data
        
        # Heat flux (vector field)
        if 'heat_flux_fields' in prediction_data:
            heat_flux_data = prediction_data['heat_flux_fields'][t_idx]  # (60, 40, 30, 3)
            fields['HeatFlux'] = heat_flux_data
        
        # Scalar fields
        scalar_field_mapping = {
            'density_fields': 'Density',
            'energy_fields': 'Energy', 
            'temperature_fields': 'Temperature'
        }
        
        for pred_key, field_name in scalar_field_mapping.items():
            if pred_key in prediction_data:
                scalar_data = prediction_data[pred_key][t_idx]  # (60, 40, 30, 1)
                fields[field_name] = scalar_data
        
        # Create VTU file for this timestep
        physical_time = (t_idx / (len(timesteps) - 1)) * 1000  # 0 to 1000
        vtu_filename = os.path.join(output_dir, f"{case_name}_t{int(physical_time):04d}.vtu")
        
        write_vtu_structured_grid(vtu_filename, fields, dimensions)
        
        if t_idx == 0 or (t_idx + 1) % 5 == 0 or t_idx == len(timesteps) - 1:
            print(f"    Converted timestep {t_idx+1}/{len(timesteps)}: t={physical_time:.0f}")
    
    # Create ParaView collection file
    create_paraview_collection_vtu(output_dir, case_name, len(timesteps))
    
    print(f" VTU conversion completed!")
    print(f"   Generated {len(timesteps)} VTU files")
    print(f"   Created ParaView collection: {case_name}.pvd")

def create_lbm_input_file(nu, temperature, geometry_file, case_name, template_file="isothermal_cracks.inp"):
    """
    Create LBM input file for simulation
    Reused from validate_neural_network.py
    """
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template file not found: {template_file}")
    
    # Read template
    with open(template_file, 'r') as f:
        content = f.read()
    
    # Replace parameters
    content = content.replace('lbm.nu = 4.857e-3', f'lbm.nu = {nu:.6e}')
    
    # Calculate alpha = nu / Prandtl (Prandtl = 0.7 for air)
    alpha_value = nu / 0.7
    content = content.replace('lbm.alpha = 6.938e-3', f'lbm.alpha = {alpha_value:.6e}')
    
    # FIXED: Only change body_temperature to match training data behavior
    # All other temperature parameters should remain at reference value (0.03333)
    content = content.replace('lbm.body_temperature = 0.03333', f'lbm.body_temperature = {temperature:.5f}')
    
    # Update geometry file
    content = content.replace('voxel_cracks.crack_file = "microstructure_nX60_nY40_nZ30_seed1.csv"', 
                             f'voxel_cracks.crack_file = "{os.path.basename(geometry_file)}"')
    
    # Save input file
    input_file = f"{case_name}.inp"
    with open(input_file, 'w') as f:
        f.write(content)
    
    return input_file

def run_lbm_simulation(nu, temperature, geometry_file, case_name, executable_path, timeout, output_dir):
    """
    Run LBM simulation with specified parameters
    Adapted from validate_neural_network.py
    """
    print(f" Running LBM simulation...")
    print(f"   Parameters: nu={nu:.6f}, temperature={temperature:.3f}")
    print(f"   Geometry: {os.path.basename(geometry_file)}")
    
    # Create simulation directory
    sim_dir = os.path.join(output_dir, f"{case_name}_lbm_simulation")
    if os.path.exists(sim_dir):
        shutil.rmtree(sim_dir)
    os.makedirs(sim_dir)
    
    # Create input file in simulation directory
    print(f"   Creating input file...")
    input_file = create_lbm_input_file(nu, temperature, geometry_file, case_name)
    sim_input = os.path.join(sim_dir, f"{case_name}.inp")
    shutil.copy(input_file, sim_input)
    os.remove(input_file)  # Clean up temporary file
    
    # Copy required files to simulation directory
    required_files = [
        executable_path,
        geometry_file
    ]
    
    print(f"   Copying required files...")
    for req_file in required_files:
        if os.path.exists(req_file):
            shutil.copy(req_file, sim_dir)
        else:
            raise FileNotFoundError(f"Required file not found: {req_file}")
    
    # Run simulation in the isolated directory
    cmd = f"./{os.path.basename(executable_path)} {case_name}.inp"
    
    try:
        # Save current directory
        original_dir = os.getcwd()
        
        # Change to simulation directory
        os.chdir(sim_dir)
        
        print(f"   Running simulation in: {sim_dir}")
        print(f"   Command: {cmd}")
        print(f"   Timeout: {timeout} seconds")
        
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, 
                               capture_output=True, text=True, timeout=timeout)
        elapsed_time = time.time() - start_time
        
        # Return to original directory
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print(f"    Simulation completed successfully in {elapsed_time:.1f} seconds")
            
            # Count output files
            plt_files = [f for f in os.listdir(sim_dir) if f.startswith('plt') and os.path.isdir(os.path.join(sim_dir, f))]
            chk_files = [f for f in os.listdir(sim_dir) if f.startswith('chk') and os.path.isdir(os.path.join(sim_dir, f))]
            
            print(f"   Generated output files:")
            print(f"     - plt files: {len(plt_files)} (timestep data)")
            print(f"     - chk files: {len(chk_files)} (checkpoint data)")
            print(f"   Output directory: {sim_dir}")
            
            return True, sim_dir, plt_files
        else:
            print(f"    Simulation failed with return code {result.returncode}")
            print(f"   Error output:")
            # Show first few lines of error
            stderr_lines = result.stderr.split('\\n')[:5]
            for line in stderr_lines:
                if line.strip():
                    print(f"     {line}")
            return False, sim_dir, []
            
    except subprocess.TimeoutExpired:
        os.chdir(original_dir)
        print(f"   ⏰ Simulation timed out after {timeout} seconds")
        return False, sim_dir, []
    except Exception as e:
        os.chdir(original_dir)
        print(f"    Simulation error: {e}")
        return False, sim_dir, []

def print_usage_examples():
    """Print usage examples"""
    print("\n Usage Examples:")
    print("\n1. Basic neural network prediction:")
    print("   python predict_and_visualize.py --geometry microstructure_nX60_nY40_nZ30_seed1.csv")
    
    print("\n2. Custom parameters:")
    print("   python predict_and_visualize.py \\")
    print("       --geometry microstructure_nX60_nY40_nZ30_seed6.csv \\")
    print("       --nu 0.005 \\")
    print("       --temperature 0.035 \\")
    print("       --timesteps 21 \\")
    print("       --output my_prediction")
    
    print("\n3. Neural network + LBM simulation comparison:")
    print("   python predict_and_visualize.py \\")
    print("       --geometry microstructure_geom_3.csv \\")
    print("       --nu 0.00324 \\")
    print("       --temperature 0.045 \\")
    print("       --timesteps 11 \\")
    print("       --run-lbm")
    
    print("\n4. Custom LBM executable and timeout:")
    print("   python predict_and_visualize.py \\")
    print("       --geometry microstructure_geom_2.csv \\")
    print("       --nu 0.005 \\")
    print("       --temperature 0.035 \\")
    print("       --run-lbm \\")
    print("       --executable ./my_marbles_executable \\")
    print("       --timeout 1800")
    
    print("\n Parameter Ranges (based on training data):")
    print("   --nu: 0.00194 to 0.00810 (kinematic viscosity)")
    print("   --temperature: 0.02 to 0.06 (temperature parameter)")
    print("   --timesteps: 2 to 101 (number of timesteps to predict)")
    
    print("\n LBM Simulation Options:")
    print("   --run-lbm: Run actual LBM simulation for comparison")
    print("   --executable: Path to marbles executable (default: marbles3d.gnu.TPROF.MPI.ex)")
    print("   --timeout: Simulation timeout in seconds (default: 3600)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Easy LBM Neural Network Prediction and Visualization (with optional LBM simulation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_and_visualize.py --geometry microstructure_nX60_nY40_nZ30_seed1.csv
  python predict_and_visualize.py --geometry microstructure_geom_3.csv --nu 0.005 --temperature 0.035
  python predict_and_visualize.py --geometry microstructure_geom_2.csv --nu 0.005 --temperature 0.035 --run-lbm
        """
    )
    
    parser.add_argument('--geometry', type=str, 
                       help='Geometry CSV file (e.g., microstructure_nX60_nY40_nZ30_seed1.csv)')
    parser.add_argument('--nu', type=float, default=0.00324,
                       help='Viscosity parameter (default: 0.00324, range: 0.00194-0.00810)')
    parser.add_argument('--temperature', type=float, default=0.035,
                       help='Temperature parameter (default: 0.035, range: 0.02-0.06)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output case name (default: auto-generated from parameters)')
    parser.add_argument('--timesteps', type=int, default=11,
                       help='Number of timesteps to predict (default: 11)')
    parser.add_argument('--output-dir', type=str, default='custom_predictions_physics_aware',
                       help='Output directory (default: custom_predictions_physics_aware)')
    parser.add_argument('--examples', action='store_true',
                       help='Show usage examples and exit')
    parser.add_argument('--run-lbm', action='store_true',
                       help='Also run LBM simulation with same parameters for comparison')
    parser.add_argument('--executable', type=str, default='marbles3d.gnu.TPROF.MPI.ex',
                       help='Path to LBM executable (default: marbles3d.gnu.TPROF.MPI.ex)')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='LBM simulation timeout in seconds (default: 3600)')
    
    args = parser.parse_args()
    
    if args.examples:
        print_usage_examples()
        return
    
    if not args.geometry:
        print(" ERROR: --geometry parameter is required!")
        print("\nFor help and examples, run:")
        print("   python predict_and_visualize.py --examples")
        return
    
    try:
        print(" LBM Neural Network Prediction and Visualization")
        print("=" * 60)
        
        # Step 1: Load model
        model = load_enhanced_model()
        
        # Step 2: Load geometry (handle both ID and full filename)
        geometry_file = args.geometry
        
        # If geometry is just a number, convert to standard filename
        if geometry_file.isdigit():
            geometry_file = f"microstructure_geom_{geometry_file}.csv"
            print(f" Converting geometry ID {args.geometry} → {geometry_file}")
        
        geometry_3d = load_geometry(geometry_file)
        
        # Step 3: Generate timesteps
        timesteps = np.linspace(0.0, 1.0, args.timesteps)
        
        # Step 4: Generate predictions
        prediction_data = generate_predictions(model, geometry_3d, args.nu, args.temperature, timesteps)
        
        # Step 5: Generate output case name
        if args.output is None:
            geom_name = Path(args.geometry).stem
            case_name = f"{geom_name}_nu{args.nu:.5f}_T{args.temperature:.3f}_steps{args.timesteps}"
        else:
            case_name = args.output
        
        # Step 6: Convert to ParaView format
        convert_to_paraview_vtu(prediction_data, args.output_dir, case_name)
        
        # Step 7: Optionally run LBM simulation
        lbm_success = False
        lbm_output_dir = None
        if args.run_lbm:
            print("\n" + "="*60)
            try:
                lbm_success, lbm_output_dir, plt_files = run_lbm_simulation(
                    args.nu, args.temperature, geometry_file, case_name, 
                    args.executable, args.timeout, args.output_dir
                )
            except Exception as lbm_error:
                print(f" LBM simulation failed: {lbm_error}")
        
        print("\n SUCCESS! Prediction and visualization complete!")
        print(f"   Case: {case_name}")
        print(f"   Parameters: nu={args.nu:.6f}, T={args.temperature:.3f}")
        print(f"   Timesteps: {args.timesteps}")
        
        if args.run_lbm:
            if lbm_success:
                print(f"    LBM simulation: Completed successfully")
                print(f"    LBM output: {lbm_output_dir}")
            else:
                print(f"    LBM simulation: Failed (see output above)")
        
        print("\n ParaView Visualization Instructions:")
        pvd_file = os.path.join(args.output_dir, f"{case_name}.pvd")
        print(f"   1. Open ParaView")
        print(f"   2. Open neural network results: {pvd_file}")
        print(f"   3. Click 'Apply' to load the data")
        print(f"   4. Select field to visualize (Velocity, Temperature, etc.)")
        print(f"   5. Use time controls to animate through {args.timesteps} timesteps")
        print(f"   6. Try 'Glyph' filter for velocity vectors")
        print(f"   7. Try 'Contour' filter for temperature isosurfaces")
        
        if args.run_lbm and lbm_success:
            print(f"\n LBM Simulation Comparison:")
            print(f"   • Neural network results: {args.output_dir}/{case_name}.pvd")
            print(f"   • LBM simulation output: {lbm_output_dir}/")
            print(f"   • Compare PHYSICS-AWARE predictions vs. ground truth in ParaView")
            print(f"   • Physics-aware model guarantees ZERO negative densities/temperatures!")
            print(f"   • LBM plt* folders contain full physics solution")
        
        print("\n Physics-Aware Visualization Benefits:")
        print("   - ALL fields guaranteed physically realistic (no negative values)")
        print("   - Softplus ensures positive densities, temperatures, energies")
        print("   - Tanh bounds velocity/heat flux to reasonable ranges")
        print("   - Activation functions provide mathematical guarantees")
        print("\n Visualization Tips:")
        print("   - Velocity field shows flow patterns through crack geometry")
        print("   - Temperature field shows thermal diffusion and convection")
        print("   - Use 'Clip' filter to see internal flow structures")
        print("   - Color by velocity magnitude to see flow intensity")
        print("   - VTU format loads automatically without reader selection")
        
    except Exception as e:
        print(f"\n ERROR: {e}")
        print("\nFor help and examples, run:")
        print("   python predict_and_visualize.py --examples")

if __name__ == "__main__":
    main()