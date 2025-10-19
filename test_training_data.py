#!/usr/bin/env python3
"""
Test script for training data generation
"""

# Import the functions from our main script
import sys
import os
sys.path.append('.')
from generate_training_data_fixed import generate_parameter_sweep, generate_all_training_data

def test_parameter_sweep():
    """Test parameter sweep generation"""
    print("Testing parameter sweep generation...")
    cases = generate_parameter_sweep("isothermal_cracks.inp")
    
    print(f"\nFirst few cases:")
    for i in range(min(3, len(cases))):
        case = cases[i]
        print(f"Case {case['case_id']}: {case['description']}")
        print(f"  nu = {case['parameters']['lbm.nu']:.6e}")
        print(f"  alpha = {case['parameters']['lbm.alpha']:.6e}")
        print(f"  temp = {case['parameters']['lbm.body_temperature']:.6e}")
        print(f"  geom file = {case['parameters']['voxel_cracks.crack_file']}")
        print()
    
    return cases

def test_small_training_data():
    """Test generating training data for just the first 5 cases"""
    print("Testing small-scale training data generation...")
    
    # Import required modules
    import shutil
    import subprocess
    
    # Create a test directory
    test_dir = "test_training_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    # Generate just the first few geometries
    print("Generating test geometries...")
    geometry_files = {}
    
    for geom_id in [1, 2]:  # Just test 2 geometries
        try:
            cmd = ["./genCracks", str(geom_id)]
            print(f"  Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                expected_output = f"results/microstructure_nX60_nY40_nZ30_seed{geom_id}.csv"
                if os.path.exists(expected_output):
                    geom_filename = f"microstructure_geom_{geom_id}.csv"
                    shutil.copy2(expected_output, geom_filename)
                    geometry_files[geom_id] = geom_filename
                    print(f"    Generated: {geom_filename}")
                    
        except Exception as e:
            print(f"    Error generating geometry {geom_id}: {e}")
    
    print(f"\nSuccessfully generated {len(geometry_files)} geometry files")
    print(f"Files created: {list(geometry_files.values())}")
    
    return geometry_files

if __name__ == "__main__":
    print("Training Data Generation Test")
    print("="*40)
    
    # Test 1: Parameter sweep
    cases = test_parameter_sweep()
    
    print("\n" + "="*40)
    
    # Test 2: Small geometry generation
    geom_files = test_small_training_data()
    
    print("\n" + "="*40)
    print(f"Test complete. Generated {len(cases)} parameter cases and {len(geom_files)} geometry files.")
    print("\nTo generate full training data (125 cases), run:")
    print("  python3 -c \"from generate_training_data_fixed import generate_all_training_data; generate_all_training_data('isothermal_cracks.inp')\"")