#!/usr/bin/env python3
"""
NDI Python Setup Helper
This script helps users set up NDI Python environment correctly.
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
import shutil

NDI_REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'ndi-python'))

def is_tool(name):
    """Check if a command-line tool exists"""
    try:
        subprocess.run([name, '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False

def check_prerequisites():
    """Check if necessary build tools are available"""
    missing_tools = []
    
    if not is_tool('cmake'):
        missing_tools.append('cmake')
    
    if not is_tool('make'):
        missing_tools.append('make')
    
    if platform.system() == 'Darwin':  # macOS
        # Check for clang/gcc
        if not is_tool('clang') and not is_tool('gcc'):
            missing_tools.append('compiler (clang or gcc)')
        
        # Check for NDI SDK
        ndi_lib_path = '/usr/local/lib/libndi.dylib'
        if not os.path.exists(ndi_lib_path):
            missing_tools.append('NDI SDK - library not found at ' + ndi_lib_path)
    
    return missing_tools

def install_ndi_python():
    """Install NDI Python directly"""
    try:
        print(f"Installing NDI Python from {NDI_REPO_PATH}...")
        
        # Mark scripts as executable
        if os.path.exists(os.path.join(NDI_REPO_PATH, 'setup.py')):
            os.chmod(os.path.join(NDI_REPO_PATH, 'setup.py'), 0o755)
        
        # Try to install the package
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-e', NDI_REPO_PATH],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print("Installation failed with the following error:")
            print(result.stderr)
            return False
        
        print("Successfully installed NDI Python!")
        return True
    except Exception as e:
        print(f"Error installing NDI Python: {e}")
        return False

def main():
    print("============================================================")
    print("NDI Python Setup Helper")
    print("============================================================")
    print(f"Python version: {sys.version}")
    print(f"Operating System: {platform.system().lower()}")
    
    # Check prerequisites
    missing = check_prerequisites()
    if missing:
        print("\nMissing required tools:")
        for tool in missing:
            print(f"  - {tool}")
        print("\nPlease install these tools before continuing.")
        
        if 'NDI SDK' in ' '.join(missing):
            print("\nTo install the NDI SDK:")
            print("1. Download from https://ndi.video/download-ndi-sdk/")
            print("2. Run the installer and follow the instructions")
        
        if 'cmake' in missing:
            print("\nTo install CMake:")
            print("  macOS: brew install cmake")
            print("  Linux: sudo apt-get install cmake")
        
        return 1
    
    # Check if NDI-Python directory exists
    if not os.path.exists(NDI_REPO_PATH):
        print(f"Error: NDI Python directory not found at {NDI_REPO_PATH}")
        return 1
    
    # Install NDI Python
    success = install_ndi_python()
    
    if success:
        print("\nSetup completed successfully!")
        print("\nTo verify the installation, run:")
        print(f"  python {os.path.join(os.path.dirname(__file__), 'test_ndi.py')}")
    else:
        print("\nSetup failed. Please check the errors above.")
        print("Falling back to simulator mode will still work correctly.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())