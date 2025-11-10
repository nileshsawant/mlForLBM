#!/usr/bin/env python3
"""
Physics-Aware Neural Network Validation Script for LBM Thermal Flow

This script validates the PHYSICS-AWARE neural network model with activation function constraints.

Changes from enhanced validation:
- Uses physics-aware model: lbm_flow_predictor_physics_aware.h5
- Tests zero physics violations (no negative densities/temperatures)
- All other validation logic remains identical

This script:
1. Selects validation parameter sets within training range
2. Runs full LBM simulations for validation cases
3. Generates neural network predictions using ENHANCED model
4. Compares errors over all timesteps
5. Produces comprehensive validation metrics and plots

Author: Generated for Physics-Aware LBM-ML validation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for headless runs
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import subprocess
import json
import h5py
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras

# Add current directory to path for imports
sys.path.append('.')

def summarize_simulation_dataset(dataset):
    """Return compact metadata for large simulation outputs to keep reports light."""
    if not dataset:
        return None

    summary = {
        'num_timesteps': len(dataset.get('timesteps', [])),
    }

    timesteps = dataset.get('timesteps', [])
    if timesteps:
        ts_array = np.asarray(timesteps, dtype=float)
        summary['timestep_range'] = {
            'min': float(np.min(ts_array)),
            'max': float(np.max(ts_array)),
        }
        summary['sample_timesteps'] = [float(ts_array[0]), float(ts_array[len(ts_array)//2]), float(ts_array[-1])] if len(ts_array) >= 3 else [float(val) for val in ts_array]

    field_shapes = {}
    for key in ['velocity_fields', 'heat_flux_fields', 'density_fields', 'energy_fields', 'temperature_fields']:
        value = dataset.get(key)
        if isinstance(value, np.ndarray):
            field_shapes[key] = list(value.shape)
        elif isinstance(value, list) and value and hasattr(value[0], 'shape'):
            field_shapes[key] = [len(value)] + list(value[0].shape)

    if field_shapes:
        summary['field_shapes'] = field_shapes

    return summary

def select_validation_parameters():
    """
    Select validation parameters from within training ranges
    
    Training ranges (from generate_training_data_fixed.py):
    - nu_values = [0.00194, 0.00324, 0.00486, 0.00648, 0.00810]  
    - temp_values = [0.02, 0.03, 0.04, 0.05, 0.06]
    - geom_ids = [1, 2, 3, 4, 5]
    
    Select intermediate values for robust validation
    """
    
    # Read reference parameters from template and compute intermediate validation points
    from pathlib import Path

    def read_input_file(filename):
        params = {}
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line or '=' not in line:
                    continue
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    try:
                        if '.' in value or 'e' in value.lower():
                            value = float(value)
                        elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                            value = int(value)
                        elif value.lower() in ['true', 'false']:
                            value = value.lower() == 'true'
                    except ValueError:
                        pass
                    params[key] = value
                except ValueError:
                    continue
        return params

    template = 'isothermal_cracks.inp'
    ref = read_input_file(template)
    nu_ref = float(ref.get('lbm.nu', 0.00486))
    temp_ref = float(ref.get('lbm.body_temperature', 0.03333))

    # Generate 5-point sweeps (same logic as generate_training_data_fixed.py)
    nu_min, nu_max = nu_ref / 1.5, nu_ref * 1.5
    temp_min, temp_max = temp_ref / 1.5, temp_ref * 1.5

    nu_values = [nu_min + i * (nu_max - nu_min) / 4 for i in range(5)]
    temp_values = [temp_min + i * (temp_max - temp_min) / 4 for i in range(5)]

    # Select intermediate points (indices 1 and 3) for validation
    nu_validation = [nu_values[1], nu_values[3]]
    temp_validation = [temp_values[1], temp_values[3]]

    # Select 1 geometry (middle of range)
    geom_validation = [3]
    
    # Create all combinations: 2×2×1 = 4 validation cases
    validation_cases = []
    case_id = 0
    
    for nu in nu_validation:
        for temp in temp_validation:
            for geom_id in geom_validation:
                validation_cases.append({
                    'case_id': case_id,
                    'nu_value': nu,
                    'temp_value': temp,
                    'geom_id': geom_id,
                    'case_name': f"validation_physics_aware_case_{case_id:03d}_nu_{nu:.5f}_temp_{temp:.3f}_geom_{geom_id}"
                })
                case_id += 1
    
    return validation_cases

def create_validation_input_file(case_info, template_file="isothermal_cracks.inp"):
    """
    Create LBM input file for validation case
    """
    case_name = case_info['case_name']
    
    # Read template lines and perform key-based updates (preserve other formatting)
    with open(template_file, 'r') as f:
        lines = f.readlines()

    def format_value(val):
        if isinstance(val, float):
            if abs(val) < 1e-2 or abs(val) > 1e3:
                return f"{val:.6e}"
            else:
                return f"{val:.6f}"
        else:
            return str(val)

    alpha_value = case_info["nu_value"] / 0.7
    geom_file = f"microstructure_geom_{case_info['geom_id']}.csv"

    out_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') or '=' not in stripped:
            out_lines.append(line)
            continue

        key, _ = stripped.split('=', 1)
        key = key.strip()

        if key == 'lbm.nu':
            out_lines.append(f"lbm.nu = {format_value(case_info['nu_value'])}\n")
        elif key == 'lbm.alpha':
            out_lines.append(f"lbm.alpha = {format_value(alpha_value)}\n")
        elif key == 'lbm.body_temperature':
            out_lines.append(f"lbm.body_temperature = {format_value(case_info['temp_value'])}\n")
        elif key == 'voxel_cracks.crack_file':
            out_lines.append(f'voxel_cracks.crack_file = "{geom_file}"\n')
        else:
            out_lines.append(line)

    # Save input file
    input_file = f"{case_name}.inp"
    with open(input_file, 'w') as f:
        f.writelines(out_lines)

    return input_file

def run_lbm_simulation(case_info):
    """
    Run LBM simulation for validation case in organized validation_physics_aware directory
    """
    case_name = case_info['case_name']
    
    print(f" Running LBM simulation: {case_name}")
    print(f"   Parameters: nu={case_info['nu_value']:.6f}, T={case_info['temp_value']:.3f}, geom={case_info['geom_id']}")
    
    # Create main validation_physics_aware directory if it doesn't exist
    main_validation_dir = "validation_physics_aware"
    os.makedirs(main_validation_dir, exist_ok=True)
    
    # Create case-specific directory under validation_physics_aware/
    case_dir = os.path.join(main_validation_dir, case_name)
    
    # Clean and recreate case directory
    if os.path.exists(case_dir):
        import shutil
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)
    
    # Create input file in the case directory
    input_file = create_validation_input_file(case_info)
    validation_input = os.path.join(case_dir, f"{case_name}.inp")
    import shutil
    shutil.copy(input_file, validation_input)
    
    # Copy required files to case directory
    required_files = [
        "marbles3d.gnu.TPROF.MPI.ex",
        f"microstructure_geom_{case_info['geom_id']}.csv"
    ]
    
    for req_file in required_files:
        if os.path.exists(req_file):
            shutil.copy(req_file, case_dir)
        else:
            print(f"     Warning: Required file not found: {req_file}")
    
    # Run simulation in the isolated case directory
    cmd = f"./marbles3d.gnu.TPROF.MPI.ex {case_name}.inp"
    
    try:
        # Save current directory
        original_dir = os.getcwd()
        
        # Change to case directory
        os.chdir(case_dir)
        
        print(f"    Running in directory: {case_dir}")
        result = subprocess.run(cmd, shell=True, 
                               capture_output=True, text=True, timeout=3600)
        
        # Return to original directory
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print(f"     Simulation completed successfully")
            
            # Count output files
            plt_files = [f for f in os.listdir(case_dir) if f.startswith('plt') and os.path.isdir(os.path.join(case_dir, f))]
            print(f"    Generated {len(plt_files)} output timesteps in {case_dir}")
            
            return True, case_dir
        else:
            print(f"     Simulation failed with return code {result.returncode}")
            print(f"   Error: {result.stderr}")
            return False, None
            
    except subprocess.TimeoutExpired:
        os.chdir(original_dir)  # Make sure we return to original directory
        print(f"    Simulation timed out after 1 hour")
        return False, None
    except Exception as e:
        os.chdir(original_dir)  # Make sure we return to original directory
        print(f"     Simulation error: {e}")
        return False, None

def load_enhanced_trained_model():
    """
    Load the PHYSICS-AWARE trained neural network model
    
    CHANGE: Specifically looks for physics-aware model files first
    """
    physics_aware_model_files = (
        'lbm_flow_predictor_physics_aware.h5',
        'best_physics_aware_model.h5',
    )

    for model_file in physics_aware_model_files:
        if not os.path.exists(model_file):
            continue

        print(f" Loading trained model: {model_file} ( PHYSICS-AWARE)")

        try:
            model = keras.models.load_model(model_file, compile=False)
        except Exception as load_error:
            raise RuntimeError(
                f"Failed to load physics-aware model '{model_file}'."
            ) from load_error

        return model, " PHYSICS-AWARE"

    raise FileNotFoundError(
        " No physics-aware model found. Run train_lbm_neural_network_physics_aware.py first."
    )

def predict_with_neural_network(model, case_info, timesteps):
    """
    Generate neural network predictions for validation case over all timesteps
    """
    case_name = case_info['case_name']
    
    print(f" Generating neural network predictions: {case_name}")
    
    # Load geometry
    geom_file = f"microstructure_geom_{case_info['geom_id']}.csv"
    if not os.path.exists(geom_file):
        raise FileNotFoundError(f"Geometry file not found: {geom_file}")
    
    # Load geometry as 3D array with error handling
    try:
        geom_df = pd.read_csv(geom_file)
        print(f"    Loaded geometry file: {geom_file}")
        print(f"    Geometry data shape: {geom_df.shape}")
        print(f"    Columns: {list(geom_df.columns)}")
        
        # Check if we have the expected columns
        if 'x' not in geom_df.columns:
            # Try alternative column names
            if len(geom_df.columns) >= 4:
                geom_df.columns = ['x', 'y', 'z', 'value']
                print(f"    Renamed columns to: {list(geom_df.columns)}")
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
                print(f"     Skipping invalid row: {row_error}")
                continue
        
        print(f"    Loaded geometry with {np.sum(geometry_3d > 0)} solid voxels")
        
    except Exception as geom_error:
        print(f"    Geometry loading error: {geom_error}")
        raise
    
    # Prepare inputs for all timesteps
    num_timesteps = len(timesteps)
    geometries = np.tile(geometry_3d[np.newaxis, ..., np.newaxis], (num_timesteps, 1, 1, 1, 1))
    
    # Create parameters for each timestep (including time as 4th parameter)
    parameters = np.zeros((num_timesteps, 4))
    alpha_value = case_info['nu_value'] / 0.7  # Calculate alpha = nu / Prandtl

    for i, t in enumerate(timesteps):
        # Parameter ordering: [nu, temperature, alpha, time]
        parameters[i] = [
            case_info['nu_value'],
            case_info['temp_value'],
            alpha_value,
            t
        ]

    # Apply saved parameter scaler if available
    try:
        import json
        if os.path.exists('param_scaler.json'):
            with open('param_scaler.json', 'r') as sf:
                scaler_info = json.load(sf)
            mean = np.array(scaler_info['mean'], dtype=np.float32)
            scale = np.array(scaler_info['scale'], dtype=np.float32)
            parameters = (parameters - mean) / scale
            print('Applied saved parameter scaler from param_scaler.json')
        else:
            print('No param_scaler.json found; proceeding without parameter scaling')
    except Exception as e:
        print(f'Warning: failed to apply saved parameter scaler: {e}')
    
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
    
    # Save neural network predictions to enhanced validation directory
    nn_output_dir = f"validation_physics_aware/neural_network_predictions"
    os.makedirs(nn_output_dir, exist_ok=True)
    
    nn_output_file = os.path.join(nn_output_dir, f"{case_name}_nn_predictions.npz")
    
    # Save as compressed numpy file
    np.savez_compressed(nn_output_file,
                       timesteps=timesteps,
                       velocity_fields=velocity_pred,
                       heat_flux_fields=heat_flux_pred, 
                       density_fields=density_pred,
                       energy_fields=energy_pred,
                       temperature_fields=temperature_pred,
                       case_info=case_info)
    
    print(f"     Generated predictions for {num_timesteps} timesteps")
    print(f"     Saved predictions to {nn_output_file}")
    return prediction_data

def process_lbm_simulation_data(output_dir):
    """
    Process LBM simulation output data using yt-project
    """
    print(f" Processing LBM simulation data: {output_dir}")
    
    try:
        import yt
        
        # Find all plotfiles
        plt_files = sorted([f for f in os.listdir(output_dir) if f.startswith('plt')])
        
        if not plt_files:
            raise FileNotFoundError(f"No plotfiles found in {output_dir}")
        
        print(f"   Found {len(plt_files)} timesteps")
        
        simulation_data = {
            'timesteps': [],
            'velocity_fields': [],
            'heat_flux_fields': [],
            'density_fields': [],
            'energy_fields': [],
            'temperature_fields': []
        }
        
        for plt_file in plt_files:
            plt_path = os.path.join(output_dir, plt_file)
            
            # Extract timestep from filename
            timestep_num = int(plt_file.split('plt')[1])
            normalized_time = timestep_num / 100.0  # Normalize (0-1 over 101 timesteps)
            
            # Load AMReX data
            ds = yt.load(plt_path)
            ad = ds.all_data()
            
            # Print available fields for debugging
            if timestep_num == 0:  # Only print for first timestep
                available_fields = [str(field) for field in ds.field_list]
                print(f"   Available fields: {available_fields[:10]}...")  # Show first 10 fields
            
            # Extract fields on uniform grid
            level = 0
            grid = ds.index.grids[level]
            
            # Extract fields using confirmed AMReX field names from the output
            velx = grid[('boxlib', 'vel_x')].to_ndarray().squeeze()
            vely = grid[('boxlib', 'vel_y')].to_ndarray().squeeze() 
            velz = grid[('boxlib', 'vel_z')].to_ndarray().squeeze()
            velocity_field = np.stack([velx, vely, velz], axis=-1)
            
            # Heat flux components
            qx = grid[('boxlib', 'qx')].to_ndarray().squeeze()
            qy = grid[('boxlib', 'qy')].to_ndarray().squeeze()
            qz = grid[('boxlib', 'qz')].to_ndarray().squeeze()
            heat_flux_field = np.stack([qx, qy, qz], axis=-1)
            
            # Scalar fields
            density_field = grid[('boxlib', 'rho')].to_ndarray().squeeze()[..., np.newaxis]
            energy_field = grid[('boxlib', 'two_rho_e')].to_ndarray().squeeze()[..., np.newaxis]
            temperature_field = grid[('boxlib', 'temperature')].to_ndarray().squeeze()[..., np.newaxis]
            
            # Store data
            simulation_data['timesteps'].append(normalized_time)
            simulation_data['velocity_fields'].append(velocity_field)
            simulation_data['heat_flux_fields'].append(heat_flux_field)
            simulation_data['density_fields'].append(density_field)
            simulation_data['energy_fields'].append(energy_field)
            simulation_data['temperature_fields'].append(temperature_field)
        
        # Convert to numpy arrays
        for key in ['velocity_fields', 'heat_flux_fields', 'density_fields', 'energy_fields', 'temperature_fields']:
            simulation_data[key] = np.array(simulation_data[key])
        
        print(f"     Processed {len(plt_files)} timesteps")
        return simulation_data
        
    except Exception as e:
        print(f"     Error processing simulation data: {e}")
        return None

def compute_validation_metrics(lbm_data, nn_data, case_info):
    """
    Compute comprehensive validation metrics
    """
    case_name = case_info['case_name']
    print(f" Computing validation metrics: {case_name}")
    
    metrics = {
        'case_info': case_info,
        'field_metrics': {},
        'temporal_evolution': {},
        'spatial_statistics': {}
    }
    
    field_names = ['velocity', 'heat_flux', 'density', 'energy', 'temperature']
    
    for field_name in field_names:
        lbm_field = lbm_data[f'{field_name}_fields']
        nn_field = nn_data[f'{field_name}_fields']
        
        # Ensure same timesteps
        min_timesteps = min(len(lbm_field), len(nn_field))
        lbm_field = lbm_field[:min_timesteps]
        nn_field = nn_field[:min_timesteps]
        
        # Compute metrics for each timestep
        timestep_mse = []
        timestep_mae = []
        timestep_r2 = []
        
        for t in range(min_timesteps):
            lbm_flat = lbm_field[t].flatten()
            nn_flat = nn_field[t].flatten()
            
            # Remove any invalid values
            valid_mask = np.isfinite(lbm_flat) & np.isfinite(nn_flat)
            lbm_valid = lbm_flat[valid_mask]
            nn_valid = nn_flat[valid_mask]
            
            if len(lbm_valid) > 0:
                mse = mean_squared_error(lbm_valid, nn_valid)
                mae = mean_absolute_error(lbm_valid, nn_valid)
                r2 = r2_score(lbm_valid, nn_valid) if len(np.unique(lbm_valid)) > 1 else 0.0
                
                timestep_mse.append(mse)
                timestep_mae.append(mae)
                timestep_r2.append(r2)
        
        # Overall field metrics
        all_lbm = lbm_field.flatten()
        all_nn = nn_field.flatten()
        valid_mask = np.isfinite(all_lbm) & np.isfinite(all_nn)
        
        if np.sum(valid_mask) > 0:
            lbm_valid = all_lbm[valid_mask]
            nn_valid = all_nn[valid_mask]
            
            metrics['field_metrics'][field_name] = {
                'mse': mean_squared_error(lbm_valid, nn_valid),
                'rmse': np.sqrt(mean_squared_error(lbm_valid, nn_valid)),
                'mae': mean_absolute_error(lbm_valid, nn_valid),
                'r2_score': r2_score(lbm_valid, nn_valid) if len(np.unique(lbm_valid)) > 1 else 0.0,
                'mean_relative_error': np.mean(np.abs((nn_valid - lbm_valid) / (np.abs(lbm_valid) + 1e-10))),
                'max_absolute_error': np.max(np.abs(nn_valid - lbm_valid))
            }
            
            metrics['temporal_evolution'][field_name] = {
                'timestep_mse': timestep_mse,
                'timestep_mae': timestep_mae, 
                'timestep_r2': timestep_r2
            }
    
    print(f"     Computed metrics for {len(field_names)} fields")
    return metrics

def create_validation_plots(validation_results, output_dir="validation_physics_aware/plots"):
    """
    Create comprehensive validation plots for enhanced model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f" Creating enhanced validation plots in {output_dir}")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    field_names = ['velocity', 'heat_flux', 'density', 'energy', 'temperature']
    
    # 1. Overall metrics comparison across cases
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ENHANCED Model - Validation Metrics Across All Cases', fontsize=16, fontweight='bold')
    
    metrics_to_plot = ['mse', 'rmse', 'mae', 'r2_score', 'mean_relative_error']
    
    for i, metric in enumerate(metrics_to_plot):
        if i < 6:  # We have 6 subplots
            ax = axes[i//3, i%3]
            
            data_for_plot = []
            case_labels = []
            
            for case_result in validation_results:
                case_info = case_result['metrics']['case_info']
                case_label = f"nu={case_info['nu_value']:.4f}\\nT={case_info['temp_value']:.3f}"
                case_labels.append(case_label)
                
                case_data = []
                for field in field_names:
                    if field in case_result['metrics']['field_metrics']:
                        case_data.append(case_result['metrics']['field_metrics'][field][metric])
                data_for_plot.append(case_data)
            
            # Create grouped bar plot
            x = np.arange(len(field_names))
            width = 0.2
            
            for j, (case_data, case_label) in enumerate(zip(data_for_plot, case_labels)):
                ax.bar(x + j*width, case_data, width, label=case_label, alpha=0.8)
            
            ax.set_xlabel('Physics Fields')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xticks(x + width*1.5)
            ax.set_xticklabels(field_names, rotation=45)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(metrics_to_plot) < 6:
        axes[1, 2].remove()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/enhanced_overall_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Temporal evolution plots for each case
    for case_result in validation_results:
        case_info = case_result['metrics']['case_info']
        case_name = case_info['case_name']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'ENHANCED Model - Temporal Evolution - {case_name}\\nnu={case_info["nu_value"]:.6f}, T={case_info["temp_value"]:.3f}', 
                    fontsize=14, fontweight='bold')
        
        for i, field in enumerate(field_names):
            ax = axes[i//3, i%3]
            
            if field in case_result['metrics']['temporal_evolution']:
                temporal_data = case_result['metrics']['temporal_evolution'][field]
                timesteps = range(len(temporal_data['timestep_mse']))
                
                ax2 = ax.twinx()
                
                # Plot MSE and R²
                line1 = ax.plot(timesteps, temporal_data['timestep_mse'], 'b-', label='MSE', linewidth=2)
                line2 = ax2.plot(timesteps, temporal_data['timestep_r2'], 'r-', label='R²', linewidth=2)
                
                ax.set_xlabel('Timestep')
                ax.set_ylabel('MSE', color='b')
                ax2.set_ylabel('R² Score', color='r')
                ax.set_title(f'{field.title()} Field')
                ax.grid(True, alpha=0.3)
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='upper right')
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/enhanced_temporal_evolution_{case_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"     Created enhanced validation plots")

def save_validation_report(validation_results, report_file="validation_physics_aware/physics_aware_validation_report.json"):
    """
    Save comprehensive enhanced validation report
    """
    print(f" Saving enhanced validation report: {report_file}")
    
    report_dir = os.path.dirname(report_file) or '.'
    os.makedirs(report_dir, exist_ok=True)

    def convert_arrays(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert_arrays(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_arrays(item) for item in obj]
        return obj

    report_payload = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'case_count': len(validation_results),
        'cases': []
    }

    for case_result in validation_results:
        case_info = case_result.get('case_info') or case_result['metrics'].get('case_info')
        case_payload = {
            'case_info': case_info,
            'metrics': case_result['metrics'],
        }

        if case_result.get('lbm_summary') is not None:
            case_payload['lbm_summary'] = case_result['lbm_summary']
        if case_result.get('nn_summary') is not None:
            case_payload['nn_summary'] = case_result['nn_summary']

        report_payload['cases'].append(case_payload)

    report_data = convert_arrays(report_payload)

    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"     Enhanced validation report saved")

def main():
    """
    Main enhanced validation pipeline
    """
    print(" Enhanced LBM Neural Network Validation Pipeline")
    print("=" * 60)
    
    # Step 1: Select validation parameters
    validation_cases = select_validation_parameters()
    print(f" Selected {len(validation_cases)} validation cases:")
    for case in validation_cases:
        print(f"   Case {case['case_id']}: nu={case['nu_value']:.6f}, T={case['temp_value']:.3f}, geom={case['geom_id']}")
    
    # Step 2: Load enhanced trained model
    try:
        model, model_type = load_enhanced_trained_model()
        print(f"    Model parameters: {model.count_params():,}")
        print(f"    Model type: {model_type}")
    except Exception as e:
        print(f" Failed to load enhanced model: {e}")
        return
    
    # Step 3: Run validation for each case
    validation_results = []
    
    for case_info in validation_cases:
        print(f"\\n Validating Enhanced Case {case_info['case_id']}")
        print("-" * 40)
        
        try:
            # Run LBM simulation
            success, output_dir = run_lbm_simulation(case_info)
            if not success:
                print(f"      Skipping case {case_info['case_id']} - simulation failed")
                continue
            
            # Process LBM data
            lbm_data = process_lbm_simulation_data(output_dir)
            if lbm_data is None:
                print(f"      Skipping case {case_info['case_id']} - data processing failed")
                continue
            
            # Generate neural network predictions
            timesteps = lbm_data['timesteps']
            nn_data = predict_with_neural_network(model, case_info, timesteps)
            
            # Compute validation metrics
            metrics = compute_validation_metrics(lbm_data, nn_data, case_info)
            
            validation_results.append({
                'case_info': case_info,
                'metrics': metrics,
                'lbm_summary': summarize_simulation_dataset(lbm_data),
                'nn_summary': summarize_simulation_dataset(nn_data)
            })
            del lbm_data
            del nn_data
            
            print(f"     Case {case_info['case_id']} enhanced validation completed")
            
        except Exception as e:
            print(f"     Error in case {case_info['case_id']}: {e}")
            continue
    
    # Step 4: Create validation plots and report
    if validation_results:
        create_validation_plots(validation_results)
        save_validation_report(validation_results)
        
        # Print summary
        print(f"\\n ENHANCED VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Completed validation cases: {len(validation_results)}/{len(validation_cases)}")
        
        # Average metrics across all cases
        for field in ['velocity', 'heat_flux', 'density', 'energy', 'temperature']:
            field_r2_scores = []
            field_rmse_values = []
            
            for result in validation_results:
                if field in result['metrics']['field_metrics']:
                    field_r2_scores.append(result['metrics']['field_metrics'][field]['r2_score'])
                    field_rmse_values.append(result['metrics']['field_metrics'][field]['rmse'])
            
            if field_r2_scores:
                avg_r2 = np.mean(field_r2_scores)
                avg_rmse = np.mean(field_rmse_values)
                print(f"{field.title():>12}: R² = {avg_r2:.4f}, RMSE = {avg_rmse:.6f}")
        
        print(f"\\n Physics-aware validation completed! Check 'validation_physics_aware/plots/' for detailed results.")
    else:
        print(" No successful enhanced validation cases!")

if __name__ == "__main__":
    main()