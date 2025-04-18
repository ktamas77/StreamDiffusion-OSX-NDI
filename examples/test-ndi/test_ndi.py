#!/usr/bin/env python3
"""
Simple test script to check if NDI SDK is properly installed and accessible.
"""

import sys
import os
import importlib.util

def check_ndi_module():
    """Check for NDI modules and print details about what was found."""
    print("Checking for NDI modules...")
    
    # List of common NDI Python bindings to check
    ndi_modules = ["NDIlib", "PyNDI", "NDI"]
    
    found_modules = []
    
    for module_name in ndi_modules:
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                # Try to import the module
                module = importlib.import_module(module_name)
                found_modules.append(module_name)
                print(f"✅ Found NDI module: {module_name}")
                
                # Try to get version or other info
                if hasattr(module, "__version__"):
                    print(f"   Version: {module.__version__}")
                if hasattr(module, "__file__"):
                    print(f"   Path: {module.__file__}")
                
                # Check for key functionality
                if hasattr(module, "initialize"):
                    print("   Has initialize() function")
                if hasattr(module, "Finder") or hasattr(module, "find_sources"):
                    print("   Has finder functionality")
                if hasattr(module, "Sender") or hasattr(module, "create_sender"):
                    print("   Has sender functionality")
                if hasattr(module, "Receiver") or hasattr(module, "create_receiver"):
                    print("   Has receiver functionality")
                    
                print()
            else:
                print(f"❌ Module {module_name} not found")
        except ImportError:
            print(f"❌ Module {module_name} found but failed to import")
        except Exception as e:
            print(f"❌ Error checking {module_name}: {e}")
    
    return found_modules

def check_ndi_sdk():
    """Check for NDI SDK installation."""
    print("Checking for NDI SDK installation...")
    
    # Common paths where NDI SDK might be installed
    possible_paths = [
        "/usr/local/lib/libndi.dylib",
        "/usr/local/lib/libndi.5.dylib",
        "/Library/NDI SDK/lib/macOS/libndi.dylib",
        "/Library/NDI SDK/lib/macOS/libndi.5.dylib",
        os.path.expanduser("~/Library/NDI SDK/lib/macOS/libndi.dylib"),
        os.path.expanduser("~/Library/NDI SDK/lib/macOS/libndi.5.dylib"),
    ]
    
    found_libs = []
    
    for path in possible_paths:
        if os.path.exists(path):
            found_libs.append(path)
            print(f"✅ Found NDI library: {path}")
    
    if not found_libs:
        print("❌ No NDI SDK libraries found in common locations")
        print("   You may need to install the NDI SDK from https://ndi.tv/sdk/")
    
    # Check environment variables
    ndi_env_vars = ["NDI_RUNTIME_DIR", "NDILIB_DIR", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"]
    
    print("\nChecking environment variables:")
    for var in ndi_env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var} = {value}")
        else:
            print(f"❌ {var} not set")
    
    return found_libs

def main():
    """Main function to run tests."""
    print("=" * 60)
    print("NDI SDK Installation Test")
    print("=" * 60)
    
    # System information
    print(f"Python version: {sys.version}")
    print(f"Operating System: {sys.platform}")
    if sys.platform == "darwin":
        try:
            from platform import mac_ver
            print(f"macOS version: {mac_ver()[0]}")
        except:
            pass
    print()
    
    # Check for NDI modules
    found_modules = check_ndi_module()
    
    # Check for NDI SDK
    found_libs = check_ndi_sdk()
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary:")
    if found_modules:
        print(f"✅ Found {len(found_modules)} NDI Python module(s): {', '.join(found_modules)}")
    else:
        print("❌ No NDI Python modules found")
    
    if found_libs:
        print(f"✅ Found {len(found_libs)} NDI library file(s)")
    else:
        print("❌ No NDI library files found")
        
    print("\nInstallation Status:")
    if found_modules and found_libs:
        print("✅ NDI SDK appears to be properly installed")
    elif found_modules:
        print("⚠️ NDI Python modules found but no library files detected")
        print("   Library files might be in non-standard locations")
    elif found_libs:
        print("⚠️ NDI library files found but no Python modules detected")
        print("   You may need to install Python bindings for NDI")
    else:
        print("❌ NDI SDK is not installed or not properly configured")
        print("   Visit https://ndi.tv/sdk/ to download and install the NDI SDK")
        print("   Then install Python bindings via pip or from source")

if __name__ == "__main__":
    main()