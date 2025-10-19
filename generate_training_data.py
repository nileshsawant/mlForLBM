#Read the input parameters file isothermal_cracks.inp

def read_input_file(filename):
    """
    Read and parse the isothermal_cracks.inp file
    
    Parameters:
    -----------
       print(f"\nGenerated {len(sweep_cases)} parameter combinations (5×5×5 = 125)")
    return sweep_cases

def write_input_file_simple(parameters, output_filename):
    """
    Write input file with updated parameters (simple version)
    
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
                    elif isinstance(value, str) and not value.startswith('"'):
                        # Quote string values if not already quoted
                        value_str = f'"{value}"'
                    else:
                        value_str = str(value)
                    
                    lines[i] = f"{key} = {value_str}\n"
            except ValueError:
                continue
    
    # Write updated file
    with open(output_filename, 'w') as f:
        f.writelines(lines)

def write_input_file_preserve_format(parameters, original_lines, output_filename):ename : str
        Path to the input file
        
    Returns:
    --------
    dict : Dictionary containing all parameters
    """
    params = {}
    
    with open(filename, 'r') as f:
        for line in f:
            # Skip empty lines and comments
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Parse parameter lines (format: key = value)
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert to appropriate type
                try:
                    # Try integer first
                    if '.' not in value and value.replace('-', '').isdigit():
                        value = int(value)
                    # Try float
                    elif value.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                        value = float(value)
                    # Try boolean
                    elif value.upper() in ['TRUE', 'FALSE']:
                        value = value.upper() == 'TRUE'
                    # Keep as string otherwise
                except ValueError:
                    pass  # Keep as string if conversion fails
                    
                params[key] = value
    
    return params

def read_input_file_preserve_format(filename):
    """
    Read the input file while preserving original formatting, comments, and line breaks
    
    Parameters:
    -----------
    filename : str
        Path to the input file
        
    Returns:
    --------
    tuple : (params_dict, original_lines)
        - params_dict: Dictionary containing parsed parameters
        - original_lines: List of original lines preserving formatting
    """
    params = {}
    original_lines = []
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            original_lines.append(line)  # Preserve original line with \n
            
            # Parse parameters while preserving original format
            stripped_line = line.strip()
            if stripped_line and not stripped_line.startswith('#') and '=' in stripped_line:
                try:
                    key, value = stripped_line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert to appropriate type
                    try:
                        # Try integer first
                        if '.' not in value and value.replace('-', '').isdigit():
                            value = int(value)
                        # Try float
                        elif value.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit():
                            value = float(value)
                        # Try boolean
                        elif value.upper() in ['TRUE', 'FALSE']:
                            value = value.upper() == 'TRUE'
                        # Keep as string otherwise
                    except ValueError:
                        pass  # Keep as string if conversion fails
                        
                    params[key] = {
                        'value': value,
                        'line_number': line_num,
                        'original_line': line.rstrip('\n')
                    }
                except ValueError:
                    # Skip malformed lines
                    continue
    
    return params, original_lines

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

def write_input_file(params, original_lines, output_filename):
    """
    Write a new input file with modified parameters while preserving format
    
    Parameters:
    -----------
    params : dict
        Dictionary of parameters to write
    original_lines : list
        Original file lines for format preservation
    output_filename : str
        Name of output file
    """
    # Read the detailed format info
    params_detailed, _ = read_input_file_preserve_format("isothermal_cracks.inp")
    
    # Create output lines
    output_lines = original_lines.copy()
    
    # Update parameter lines
    for key, new_value in params.items():
        if key in params_detailed:
            line_num = params_detailed[key]['line_number'] - 1  # Convert to 0-based index
            
            # Format the new value appropriately
            if isinstance(new_value, float):
                if new_value < 1e-3 or new_value > 1e3:
                    value_str = f"{new_value:.6e}"
                else:
                    value_str = f"{new_value:.6f}"
            else:
                value_str = str(new_value)
            
            # Replace the line while preserving format
            original_line = params_detailed[key]['original_line']
            if '=' in original_line:
                key_part, _ = original_line.split('=', 1)
                new_line = f"{key_part}= {value_str}\n"
                output_lines[line_num] = new_line
    
    # Write the new file
    with open(output_filename, 'w') as f:
        f.writelines(output_lines)
    
    print(f"Written input file: {output_filename}")

def generate_all_sweep_files(input_filename="isothermal_cracks.inp", output_dir="sweep_cases"):
    """
    Generate all input files for the parameter sweep
    
    Parameters:
    -----------
    input_filename : str
        Reference input file
    output_dir : str
        Directory to store sweep cases
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get original file format
    _, original_lines = read_input_file_preserve_format(input_filename)
    
    # Generate sweep cases
    sweep_cases = generate_parameter_sweep(input_filename)
    
    print(f"\nGenerating {len(sweep_cases)} input files in '{output_dir}/':")
    
    for case in sweep_cases:
        output_filename = f"{output_dir}/case_{case['case_id']:02d}_{case['description']}.inp"
        write_input_file(case['parameters'], original_lines, output_filename)
        
        print(f"  Case {case['case_id']:2d}: nu={case['parameters']['lbm.nu']:.6e}, "
              f"alpha={case['parameters']['lbm.alpha']:.6e}, "
              f"temp={case['parameters']['lbm.body_temperature']:.6e}")
    
    return sweep_cases

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
                # Look for generated CSV file (assuming genCracks creates microstructure_*.csv)
                # You may need to adjust this based on actual genCracks output naming
                default_output = "microstructure_nX60_nY40_nZ30_seed1.csv"
                if os.path.exists(default_output):
                    # Copy to our naming convention
                    shutil.copy2(default_output, geom_filename)
                    geometry_files[geom_id] = geom_filename
                    print(f"    Generated: {geom_filename}")
                else:
                    print(f"    Warning: Expected output {default_output} not found for genCracks {geom_id}")
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
            write_input_file_simple(case['parameters'], input_file)
            
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

# Example usage:
if __name__ == "__main__":
    # Simple parameter extraction
    input_params = read_input_file("isothermal_cracks.inp")
    print("Input parameters:")
    for key, value in input_params.items():
        print(f"  {key} = {value}")
    
    print("\n" + "="*50 + "\n")
    
    # Format-preserving read
    params_detailed, original_lines = read_input_file_preserve_format("isothermal_cracks.inp")
    print("Parameters with original formatting:")
    for key, info in params_detailed.items():
        print(f"  Line {info['line_number']}: {info['original_line']}")
        print(f"    Parsed value: {info['value']} (type: {type(info['value']).__name__})")
    
    print(f"\nTotal lines in file: {len(original_lines)}")
    print("First 5 lines of original file:")
    for i, line in enumerate(original_lines[:5]):
        print(f"  {i+1}: {repr(line)}")  # repr shows \n characters


