#!/usr/bin/env python3
"""
Cleanup script for training data directories
"""

import os
import glob
import shutil

def cleanup_simulation_outputs(training_data_dir="training_data"):
    """
    Remove all simulation output files from training data directories
    Keeps only the input files, geometry files, and executables
    """
    print("Cleaning up simulation output files...")
    print("=" * 50)
    
    # Find all case directories
    case_pattern = os.path.join(training_data_dir, "case_*")
    case_dirs = sorted(glob.glob(case_pattern))
    
    if not case_dirs:
        print(f" No case directories found in {training_data_dir}")
        return
    
    print(f"Found {len(case_dirs)} case directories to clean")
    
    total_files_removed = 0
    total_size_freed = 0
    
    for i, case_dir in enumerate(case_dirs):
        case_name = os.path.basename(case_dir)
        
        # Files to remove (simulation outputs)
        patterns_to_remove = [
            "plt*",
            "chk*", 
            "Backtrace.*",
            "*.old.*",
            "Header",
            "Level_*"
        ]
        
        files_removed = 0
        size_freed = 0
        
        for pattern in patterns_to_remove:
            files = glob.glob(os.path.join(case_dir, pattern))
            for file_path in files:
                try:
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        files_removed += 1
                        size_freed += size
                    elif os.path.isdir(file_path):
                        # For directories like Level_0, chk00000, etc.
                        dir_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                                     for dirpath, dirnames, filenames in os.walk(file_path)
                                     for filename in filenames)
                        shutil.rmtree(file_path)
                        files_removed += 1
                        size_freed += dir_size
                except Exception as e:
                    print(f"    Warning: Could not remove {file_path}: {e}")
        
        if files_removed > 0:
            print(f"  {case_name}: Removed {files_removed} items ({size_freed/1024/1024:.1f} MB)")
        
        total_files_removed += files_removed
        total_size_freed += size_freed
        
        # Progress update every 25 cases
        if (i + 1) % 25 == 0:
            print(f"    Progress: {i+1}/{len(case_dirs)} directories cleaned...")
    
    print(f"\n" + "=" * 50)
    print(f"Cleanup Complete!")
    print(f"  Directories processed: {len(case_dirs)}")
    print(f"  Total items removed: {total_files_removed}")
    print(f"  Total space freed: {total_size_freed/1024/1024:.1f} MB")
    
    # Verify what's left in first case
    if case_dirs:
        first_case = case_dirs[0]
        remaining_files = os.listdir(first_case)
        print(f"\nRemaining files in {os.path.basename(first_case)}:")
        for file in sorted(remaining_files):
            file_path = os.path.join(first_case, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"   {file} ({size:,} bytes)")
            else:
                print(f"   {file}/ (directory)")

def verify_clean_state(training_data_dir="training_data"):
    """
    Verify that all case directories are in clean state
    """
    print(f"\nVerifying clean state...")
    
    case_pattern = os.path.join(training_data_dir, "case_*")
    case_dirs = sorted(glob.glob(case_pattern))
    
    expected_files = [
        "isothermal_cracks.inp",
        "marbles3d.gnu.TPROF.MPI.ex"
    ]
    
    expected_patterns = [
        "microstructure_geom_*.csv"
    ]
    
    clean_cases = 0
    
    for case_dir in case_dirs[:5]:  # Check first 5 cases
        case_name = os.path.basename(case_dir)
        files = os.listdir(case_dir)
        
        # Check for unexpected output files
        unexpected_files = []
        for file in files:
            if (file not in expected_files and 
                not any(glob.fnmatch.fnmatch(file, pattern) for pattern in expected_patterns) and
                not file.startswith('microstructure_geom_')):
                unexpected_files.append(file)
        
        if not unexpected_files:
            clean_cases += 1
            print(f"   {case_name}: Clean")
        else:
            print(f"    {case_name}: Has unexpected files: {unexpected_files}")
    
    if clean_cases == min(5, len(case_dirs)):
        print(f"   All checked cases are clean and ready for batch simulation")
    else:
        print(f"    Some cases may need additional cleanup")

if __name__ == "__main__":
    print("Training Data Cleanup")
    print("=" * 50)
    
    # Clean up all simulation outputs
    cleanup_simulation_outputs()
    
    # Verify clean state
    verify_clean_state()
    
    print(f"\n Cleanup complete! All case directories are ready for batch simulation.")