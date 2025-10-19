#!/usr/bin/env python3
"""
Training Data Generator for Marbles LBM Simulations

This script generates parameter sweeps for machine learning training data.
It creates 125 cases (5 viscosities × 5 temperatures × 5 geometries) for 
thermal lattice Boltzmann method simulations.
"""

def read_input_file(filename):
    """
    Read AMReX input file and extract parameters
    
    Parameters:
    -----------
    filename : str
        Path to the input file
        
    Returns:
    --------
    dict : Dictionary of parameter name -> value
    """
    params = {}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('#') or not line or '=' not in line:
                continue
                
            try:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                    
                # Try to convert to appropriate type
                try:
                    if '.' in value or 'e' in value.lower():
                        value = float(value)
                    elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                        value = int(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    # Keep as string otherwise
                except ValueError:
                    pass  # Keep as string if conversion fails
                    
                params[key] = value
            except ValueError:
                # Skip malformed lines
                continue
    
    return params

def generate_parameter_sweep(filename):
    """
    Generate parameter sweep based on reference values in isothermal_cracks.inp
    
    Sweeps:
    - 5 viscosities: lbm.nu/1.5 to lbm.nu*1.5  
    - 5 body temperatures: lbm.body_temperature/1.5 to lbm.body_temperature*1.5
    - 5 geometries: genCracks 1, genCracks 2, genCracks 3, genCracks 4, genCracks 5
    - lbm.alpha = lbm.nu/0.7 (always coupled to viscosity)
    
    Parameters:
    -----------
    filename : str
        Path to the input file
        
    Returns:
    --------
    list : List of parameter dictionaries for each sweep case
    """
    # Read reference parameters
    ref_params = read_input_file(filename)
    
    # Get reference values
    nu_ref = ref_params['lbm.nu']
    temp_ref = ref_params['lbm.body_temperature']
    
    print(f"Reference values:")
    print(f"  lbm.nu = {nu_ref}")
    print(f"  lbm.body_temperature = {temp_ref}")
    print(f"  lbm.alpha (calculated) = {nu_ref/0.7:.6e}")
    
    # Generate sweep ranges (using basic Python, no numpy)
    nu_values = []
    temp_values = []
    geom_ids = [1, 2, 3, 4, 5]
    
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
    
    print(f"\nSweep ranges:")
    print(f"  nu values: {[f'{v:.6e}' for v in nu_values]}")
    print(f"  temperature values: {[f'{v:.6e}' for v in temp_values]}")
    print(f"  geometry IDs: {geom_ids}")
    
    # Generate all combinations
    sweep_cases = []
    case_id = 0
    
    for i, nu in enumerate(nu_values):
        for j, temp in enumerate(temp_values):
            for k, geom_id in enumerate(geom_ids):
                alpha = nu / 0.7  # Always coupled to viscosity
                
                case_params = ref_params.copy()
                case_params['lbm.nu'] = nu
                case_params['lbm.alpha'] = alpha
                case_params['lbm.body_temperature'] = temp
                case_params['voxel_cracks.crack_file'] = f"microstructure_geom_{geom_id}.csv"
                
                sweep_cases.append({
                    'case_id': case_id,
                    'nu_index': i,
                    'temp_index': j,
                    'geom_index': k,
                    'geom_id': geom_id,
                    'parameters': case_params,
                    'description': f"nu_{i}_temp_{j}_geom_{geom_id}"
                })
                case_id += 1
    
    print(f"\nGenerated {len(sweep_cases)} parameter combinations (5×5×5 = 125)")
    return sweep_cases

def write_input_file(parameters, output_filename):
    """
    Write input file with updated parameters
    
    Parameters:
    -----------
    parameters : dict
        Parameter dictionary from read_input_file
    output_filename : str
        Output file path
    """
    # Read original file for structure
    with open("isothermal_cracks.inp", 'r') as f:
        lines = f.readlines()
    
    # Update lines with new parameter values
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if '=' in line_stripped and not line_stripped.startswith('#'):
            try:
                key, _ = line_stripped.split('=', 1)
                key = key.strip()
                
                if key in parameters:
                    # Format the new value
                    value = parameters[key]
                    if isinstance(value, float):
                        if abs(value) < 1e-2 or abs(value) > 1e3:
                            # Use scientific notation for very small/large numbers
                            value_str = f"{value:.6e}"
                        else:
                            value_str = f"{value:.6f}"
                    elif isinstance(value, str):
                        # Check if it's a multi-value parameter (contains spaces)
                        if ' ' in value and not value.startswith('"'):
                            # Multi-value parameters shouldn't be quoted
                            value_str = value
                        elif not value.startswith('"'):
                            # Single string values should be quoted
                            value_str = f'"{value}"'
                        else:
                            # Already quoted
                            value_str = value
                    else:
                        value_str = str(value)
                    
                    lines[i] = f"{key} = {value_str}\n"
            except ValueError:
                continue
    
    # Write updated file
    with open(output_filename, 'w') as f:
        f.writelines(lines)

def generate_all_training_data(input_filename, output_dir="training_data", executable_path="marbles3d.gnu.TPROF.MPI.ex"):
    """
    Generate complete training data structure with:
    - 125 cases (5 viscosities × 5 temperatures × 5 geometries)
    - Each case has: input file (.inp), geometry file (.csv), executable copy
    
    Parameters:
    -----------
    input_filename : str
        Path to the reference input file
    output_dir : str  
        Directory to create training data in
    executable_path : str
        Path to the marbles executable to copy
    """
    import os
    import shutil
    import subprocess
    
    # Generate parameter sweep
    sweep_cases = generate_parameter_sweep(input_filename)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating training data structure with {len(sweep_cases)} cases in {output_dir}/")
    
    # Check if executable exists
    if not os.path.exists(executable_path):
        print(f"Warning: Executable not found at {executable_path}")
        print("Training data will be generated without executable copies")
        copy_executable = False
    else:
        copy_executable = True
        print(f"Will copy executable from: {executable_path}")
    
    # Generate geometries first (5 total)
    print(f"\nGenerating 5 geometry files using genCracks...")
    geometry_files = {}
    
    for geom_id in [1, 2, 3, 4, 5]:
        geom_filename = f"microstructure_geom_{geom_id}.csv"
        
        # Run genCracks to generate geometry
        try:
            cmd = ["./genCracks", str(geom_id)]
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                # genCracks creates files in results/ directory with naming pattern:
                # microstructure_nX60_nY40_nZ30_seed{geom_id}.csv
                expected_output = f"results/microstructure_nX60_nY40_nZ30_seed{geom_id}.csv"
                if os.path.exists(expected_output):
                    # Copy to our naming convention
                    shutil.copy2(expected_output, geom_filename)
                    geometry_files[geom_id] = geom_filename
                    print(f"    Generated: {geom_filename}")
                else:
                    print(f"    Warning: Expected output {expected_output} not found for genCracks {geom_id}")
            else:
                print(f"    Error running genCracks {geom_id}: {result.stderr}")
                
        except FileNotFoundError:
            print(f"    Error: genCracks executable not found")
            break
        except Exception as e:
            print(f"    Error generating geometry {geom_id}: {e}")
    
    # Generate case directories
    print(f"\nGenerating {len(sweep_cases)} case directories...")
    
    successful_cases = 0
    for case in sweep_cases:
        case_dir = os.path.join(output_dir, f"case_{case['case_id']:03d}_{case['description']}")
        os.makedirs(case_dir, exist_ok=True)
        
        try:
            # Write input file
            input_file = os.path.join(case_dir, "isothermal_cracks.inp")
            write_input_file(case['parameters'], input_file)
            
            # Copy geometry file
            geom_id = case['geom_id']
            if geom_id in geometry_files:
                geom_source = geometry_files[geom_id]
                geom_dest = os.path.join(case_dir, f"microstructure_geom_{geom_id}.csv")
                shutil.copy2(geom_source, geom_dest)
            else:
                print(f"    Warning: Geometry file for geom_id {geom_id} not available")
            
            # Copy executable
            if copy_executable:
                exec_dest = os.path.join(case_dir, os.path.basename(executable_path))
                shutil.copy2(executable_path, exec_dest)
                # Make executable
                os.chmod(exec_dest, 0o755)
            
            successful_cases += 1
            
            if case['case_id'] % 25 == 0:  # Progress update every 25 cases
                print(f"  Completed {case['case_id'] + 1}/{len(sweep_cases)} cases...")
                
        except Exception as e:
            print(f"    Error generating case {case['case_id']}: {e}")
    
    print(f"\nTraining data generation complete!")
    print(f"Successfully generated: {successful_cases}/{len(sweep_cases)} cases")
    print(f"Each case directory contains:")
    print(f"  - isothermal_cracks.inp (simulation parameters)")
    print(f"  - microstructure_geom_*.csv (geometry file)")
    if copy_executable:
        print(f"  - {os.path.basename(executable_path)} (executable copy)")
    
    return successful_cases

def run_all_simulations(training_data_dir="training_data"):
    """
    Run simulations in all training data case directories
    Uses the exact command: marbles3d.gnu.TPROF.MPI.ex isothermal_cracks.inp
    
    Parameters:
    -----------
    training_data_dir : str
        Directory containing all case folders
    """
    import os
    import subprocess
    import glob
    import time
    
    # Find all case directories
    case_pattern = os.path.join(training_data_dir, "case_*")
    case_dirs = sorted(glob.glob(case_pattern))
    
    print(f"Batch Simulation Execution")
    print(f"=" * 50)
    print(f"Found {len(case_dirs)} case directories to run")
    
    if not case_dirs:
        print(f" No case directories found in {training_data_dir}")
        return 0
    
    # Check if first case has the executable
    test_case = case_dirs[0]
    executable_path = os.path.join(test_case, "marbles3d.gnu.TPROF.MPI.ex")
    input_file = os.path.join(test_case, "isothermal_cracks.inp")
    
    if not os.path.exists(executable_path):
        print(f" Executable not found in {test_case}")
        return 0
    
    if not os.path.exists(input_file):
        print(f" Input file not found in {test_case}")
        return 0
    
    print(f" Executable and input files verified")
    print(f"Command: ./marbles3d.gnu.TPROF.MPI.ex isothermal_cracks.inp")
    
    successful_runs = 0
    failed_runs = 0
    start_time = time.time()
    
    print(f"\nStarting simulations...")
    print(f"")
    
    for i, case_dir in enumerate(case_dirs):
        case_name = os.path.basename(case_dir)
        
        print(f"[{i+1:3d}/{len(case_dirs)}] Running {case_name}...")
        
        try:
            # Use the exact command you specified
            cmd = ["./marbles3d.gnu.TPROF.MPI.ex", "isothermal_cracks.inp"]
            
            # Run simulation in the case directory
            result = subprocess.run(
                cmd,
                cwd=case_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per simulation
            )
            
            if result.returncode == 0:
                successful_runs += 1
                print(f"     Completed successfully")
                
                # Check if output files were created
                plt_files = glob.glob(os.path.join(case_dir, "plt*"))
                if plt_files:
                    print(f"     Generated {len(plt_files)} output files")
                else:
                    print(f"      No plt* output files found")
                    
            else:
                failed_runs += 1
                print(f"     Failed (exit code: {result.returncode})")
                # Show first few lines of error
                stderr_lines = result.stderr.split('\n')[:3]
                for line in stderr_lines:
                    if line.strip():
                        print(f"    Error: {line.strip()}")
                
        except subprocess.TimeoutExpired:
            failed_runs += 1
            print(f"     Timeout (exceeded 10 minutes)")
            
        except Exception as e:
            failed_runs += 1
            print(f"     Error: {e}")
        
        # Progress update every 10 cases
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(case_dirs) - i - 1)
            print(f"     Progress: {i+1}/{len(case_dirs)} | "
                  f"Success: {successful_runs} | Failed: {failed_runs} | "
                  f"ETA: {remaining/60:.1f} min")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n" + "=" * 50)
    print(f"Batch Simulation Complete!")
    print(f"  Total cases: {len(case_dirs)}")
    print(f"  Successful: {successful_runs}")
    print(f"  Failed: {failed_runs}")
    print(f"  Success rate: {successful_runs/len(case_dirs)*100:.1f}%")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Average time per case: {total_time/len(case_dirs):.1f} seconds")
    
    return successful_runs

# Example usage:
if __name__ == "__main__":
    print("Training Data Generator for Marbles LBM Simulations")
    print("="*50)
    
    # Test parameter reading
    input_params = read_input_file("isothermal_cracks.inp")
    print("Key parameters from isothermal_cracks.inp:")
    for key in ['lbm.nu', 'lbm.alpha', 'lbm.body_temperature', 'voxel_cracks.crack_file']:
        if key in input_params:
            print(f"  {key} = {input_params[key]}")
    
    print("\n" + "="*50)
    print("To generate full training data, run:")
    print("  generate_all_training_data('isothermal_cracks.inp')")
    print("\nTo run all simulations, run:")
    print("  run_all_simulations()")
    print("\nThis will create 125 simulation cases with parameter sweeps and geometries.")