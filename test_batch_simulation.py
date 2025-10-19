#!/usr/bin/env python3
"""
Test batch simulation execution on a small subset
"""

import os
import sys
import glob
import subprocess
import time

def test_single_simulation():
    """Test running a single case to make sure everything works"""
    print("Testing Single Simulation")
    print("=" * 40)
    
    # Find the first case directory
    case_dirs = sorted(glob.glob("training_data/case_*"))
    if not case_dirs:
        print(" No case directories found!")
        return False
    
    test_case = case_dirs[0]
    case_name = os.path.basename(test_case)
    
    print(f"Testing case: {case_name}")
    
    # Check files exist
    executable = os.path.join(test_case, "marbles3d.gnu.TPROF.MPI.ex")
    input_file = os.path.join(test_case, "isothermal_cracks.inp")
    
    if not os.path.exists(executable):
        print(f" Executable not found: {executable}")
        return False
    
    if not os.path.exists(input_file):
        print(f" Input file not found: {input_file}")
        return False
    
    print(f" Files verified")
    
    # Test the exact command you specified
    cmd = ["./marbles3d.gnu.TPROF.MPI.ex", "isothermal_cracks.inp"]
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {test_case}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            cwd=test_case,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout for test
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f" Simulation completed successfully in {elapsed:.1f} seconds")
            
            # Check for output files
            plt_files = glob.glob(os.path.join(test_case, "plt*"))
            chk_files = glob.glob(os.path.join(test_case, "chk*"))
            
            print(f"Output files generated:")
            print(f"  - plt files: {len(plt_files)}")
            print(f"  - chk files: {len(chk_files)}")
            
            if plt_files:
                print(f"  Latest plt file: {os.path.basename(plt_files[-1])}")
            
            return True
            
        else:
            print(f" Simulation failed (exit code: {result.returncode})")
            print(f"stdout: {result.stdout[:500]}")
            print(f"stderr: {result.stderr[:500]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f" Simulation timed out (>5 minutes)")
        return False
        
    except Exception as e:
        print(f" Error running simulation: {e}")
        return False

def test_small_batch():
    """Test running the first 3 cases"""
    print("\nTesting Small Batch (3 cases)")
    print("=" * 40)
    
    # Import the function from our main script
    sys.path.append('.')
    from generate_training_data_fixed import run_all_simulations
    
    # Create a temporary directory with just 3 cases for testing
    import shutil
    
    test_dir = "test_batch"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)
    
    # Copy first 3 cases
    case_dirs = sorted(glob.glob("training_data/case_*"))[:3]
    
    for case_dir in case_dirs:
        case_name = os.path.basename(case_dir)
        dest_dir = os.path.join(test_dir, case_name)
        shutil.copytree(case_dir, dest_dir)
        print(f"  Copied {case_name}")
    
    print(f"\nRunning batch simulation on {len(case_dirs)} cases...")
    
    # Run the batch simulation
    successful = run_all_simulations(test_dir)
    
    print(f"\nTest batch completed: {successful}/{len(case_dirs)} successful")
    
    # Clean up
    shutil.rmtree(test_dir)
    
    return successful == len(case_dirs)

if __name__ == "__main__":
    print("Batch Simulation Testing")
    print("=" * 50)
    
    # Test 1: Single simulation
    single_success = test_single_simulation()
    
    if single_success:
        # Test 2: Small batch
        batch_success = test_small_batch()
        
        if batch_success:
            print(f"\n All tests passed!")
            print(f"Ready to run full batch simulation with:")
            print(f"  python3 -c \"from generate_training_data_fixed import run_all_simulations; run_all_simulations()\"")
        else:
            print(f"\n Batch test failed")
    else:
        print(f"\n Single simulation test failed")