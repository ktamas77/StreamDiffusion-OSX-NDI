#!/usr/bin/env python3
"""
Script to test if NDI Python bindings are correctly installed.
"""
import sys
print(f"Python version: {sys.version}")

try:
    import NDIlib as ndi
    print("NDIlib is correctly installed!")
    
    # Try to initialize NDI
    if ndi.initialize():
        print("NDI successfully initialized")
        
        # Try finding sources
        finder = ndi.Finder()
        finder.wait_for_sources(1000)  # Wait for 1 second
        sources = finder.get_sources()
        
        if sources:
            print(f"Found {len(sources)} NDI sources:")
            for i, source in enumerate(sources):
                print(f"  Source {i+1}: {source.name}")
        else:
            print("No NDI sources found (this is normal if no NDI sources are broadcasting)")
            
        # Clean up
        ndi.terminate()
    else:
        print("Failed to initialize NDI")
        
except ImportError:
    print("NDIlib module not found. The installation failed.")
    sys.exit(1)
