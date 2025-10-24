#!/usr/bin/env python3
"""
Installation script for SuGaR CUDA extensions.
This script compiles and sets up the Python path for the CUDA extensions.
"""

import os
import sys
import subprocess
import site

def run_command(cmd, cwd=None):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("Installing SuGaR CUDA extensions...")
    
    # Get the SuGaR root directory
    sugar_root = os.path.dirname(os.path.abspath(__file__))
    
    # Paths to the CUDA extension directories
    diff_gaussian_path = os.path.join(sugar_root, "gaussian_splatting", "submodules", "diff-gaussian-rasterization")
    simple_knn_path = os.path.join(sugar_root, "gaussian_splatting", "submodules", "simple-knn")
    
    print(f"SuGaR root: {sugar_root}")
    print(f"diff-gaussian-rasterization: {diff_gaussian_path}")
    print(f"simple-knn: {simple_knn_path}")
    
    # Check if directories exist
    if not os.path.exists(diff_gaussian_path):
        print(f"Error: {diff_gaussian_path} not found!")
        return False
    if not os.path.exists(simple_knn_path):
        print(f"Error: {simple_knn_path} not found!")
        return False
    
    # Compile diff-gaussian-rasterization
    print("\n1. Compiling diff-gaussian-rasterization...")
    if not run_command("python setup.py build_ext --inplace", cwd=diff_gaussian_path):
        print("Failed to compile diff-gaussian-rasterization")
        return False
    
    # Compile simple-knn
    print("\n2. Compiling simple-knn...")
    if not run_command("python setup.py build_ext --inplace", cwd=simple_knn_path):
        print("Failed to compile simple-knn")
        return False
    
    # Create a .pth file to add the paths to Python
    print("\n3. Setting up Python path...")
    
    # Get site-packages directory
    site_packages = site.getsitepackages()[0]
    pth_file = os.path.join(site_packages, "sugar_cuda_extensions.pth")
    
    with open(pth_file, 'w') as f:
        f.write(f"{diff_gaussian_path}\n")
        f.write(f"{simple_knn_path}\n")
    
    print(f"Created .pth file: {pth_file}")
    
    # Test the installation
    print("\n4. Testing installation...")
    test_cmd = f"""
import sys
sys.path.insert(0, '{diff_gaussian_path}')
sys.path.insert(0, '{simple_knn_path}')
try:
    import diff_gaussian_rasterization
    print('✓ diff_gaussian_rasterization imported successfully')
except ImportError as e:
    print('✗ Failed to import diff_gaussian_rasterization:', str(e))

try:
    import simple_knn
    print('✓ simple_knn imported successfully')
except ImportError as e:
    print('✗ Failed to import simple_knn:', str(e))
"""
    
    if run_command(f"python -c \"{test_cmd}\""):
        print("\n🎉 Installation completed successfully!")
        print("\nBoth CUDA extensions are now available in your Python environment.")
        print("You can import them with:")
        print("  import diff_gaussian_rasterization")
        print("  import simple_knn")
        return True
    else:
        print("\n❌ Installation completed but testing failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
