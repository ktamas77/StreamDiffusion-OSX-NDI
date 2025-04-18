#!/usr/bin/env python3
"""
Script to test if NDI Python bindings are correctly installed.
"""
import sys
import time
print(f"Python version: {sys.version}")

try:
    import NDIlib as ndi
    print("NDIlib is correctly installed!")
    
    # Try to initialize NDI
    if ndi.initialize():
        print("NDI successfully initialized")
        
        # Check for available attributes and methods
        print("\nAvailable NDI attributes and methods:")
        for attr in dir(ndi):
            if not attr.startswith('_'):  # Skip private attributes
                print(f"  - {attr}")
        
        # Try to find sources
        if hasattr(ndi, 'find_create_v2'):
            print("\nSearching for NDI sources using find_create_v2...")
            finder = ndi.find_create_v2()
            if finder:
                # Wait for sources to be discovered
                ndi.find_wait_for_sources(finder, 1000)  # 1 second timeout
                sources = ndi.find_get_current_sources(finder)
                
                if sources:
                    print(f"Found {len(sources)} NDI sources:")
                    for i, source in enumerate(sources):
                        # Try different attribute names
                        source_name = None
                        if hasattr(source, 'name'):
                            source_name = source.name
                        elif hasattr(source, 'ndi_name'):
                            source_name = source.ndi_name
                        else:
                            source_name = str(source)
                        print(f"  Source {i+1}: {source_name}")
                else:
                    print("No NDI sources found (this is normal if no NDI sources are broadcasting)")
                
                # Clean up
                ndi.find_destroy(finder)
            else:
                print("Failed to create NDI finder")
        else:
            print("\nCould not find appropriate NDI source discovery method")
            print("This could be due to differences in NDI Python binding implementations")
        
        # Clean up - use destroy instead of terminate if terminate is not available
        if hasattr(ndi, 'destroy'):
            ndi.destroy()
        
        print("\nNDI test completed successfully")
    else:
        print("Failed to initialize NDI")
        
except ImportError:
    print("NDIlib module not found. The installation failed.")
    sys.exit(1)
except Exception as e:
    print(f"Error testing NDI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)