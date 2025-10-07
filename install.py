#!/usr/bin/env python3
"""
Installation script for stim_transformations library.

This script provides an easy way to install the library and its dependencies.
Run with: python install.py
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        print(f"  Current version: {sys.version}")
        return False
    else:
        print(f"✓ Python version {sys.version.split()[0]} is compatible")
        return True


def install_dependencies():
    """Install required dependencies."""
    print("\n" + "="*50)
    print("INSTALLING DEPENDENCIES")
    print("="*50)
    
    # Install basic dependencies
    basic_deps = [
        "numpy>=1.21.0",
        "opencv-python>=4.5.0", 
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0"
    ]
    
    for dep in basic_deps:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            return False
    
    # Install PyTorch (CPU version by default)
    pytorch_cmd = "pip install torch torchvision"
    if not run_command(pytorch_cmd, "Installing PyTorch (CPU version)"):
        print("Warning: PyTorch installation failed. You may need to install it manually.")
        print("For GPU support, visit: https://pytorch.org/get-started/locally/")
    
    # Install plenoptic
    if not run_command("pip install plenoptic", "Installing plenoptic"):
        print("Warning: plenoptic installation failed. You may need to install it manually.")
        print("Try: pip install plenoptic")
    
    return True


def install_package():
    """Install the package in development mode."""
    print("\n" + "="*50)
    print("INSTALLING PACKAGE")
    print("="*50)
    
    # Change to the package directory
    package_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(package_dir)
    
    # Install in development mode
    if not run_command("pip install -e .", "Installing stim_transformations in development mode"):
        return False
    
    return True


def verify_installation():
    """Verify the installation works."""
    print("\n" + "="*50)
    print("VERIFYING INSTALLATION")
    print("="*50)
    
    try:
        # Test basic import
        import stim_transformations as stf
        print("✓ stim_transformations imported successfully")
        
        # Test basic functionality with a simple image
        import numpy as np
        test_img = np.zeros((50, 50), dtype=np.uint8)
        test_img[20:30, 20:30] = 255
        
        result = stf.setup(test_img, 'center')
        print("✓ Basic functionality test passed")
        
        return True
    except Exception as e:
        print(f"✗ Installation verification failed: {e}")
        return False


def main():
    """Main installation function."""
    print("Stimulus Transformations Library - Installation Script")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Dependency installation failed!")
        print("Please install dependencies manually:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Install package
    if not install_package():
        print("\n✗ Package installation failed!")
        sys.exit(1)
    
    # Verify installation
    verify_installation()
    
    print("\n" + "="*60)
    print("INSTALLATION COMPLETED!")
    print("="*60)
    print("You can now use the library:")
    print("  import stim_transformations as stf")
    print("\nTry the tutorial:")
    print("  jupyter notebook tutorial/stim_transformations_tutorial.ipynb")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
