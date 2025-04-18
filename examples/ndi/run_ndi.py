#!/usr/bin/env python3
"""
Enhanced NDI example for StreamDiffusion on macOS.
Uses the NDI Python bindings installed in the ndi-env environment.
"""

import os
import sys
import argparse
import subprocess

# Get the absolute path to the StreamDiffusion-OSX directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_ndi_sources():
    """Get a list of available NDI sources using the test script."""
    test_script = os.path.join(repo_root, "test_ndi_installation.py")
    
    # Get the path to Python in the ndi-env
    python_path = os.path.expanduser("~/.pyenv/versions/3.10.12/envs/ndi-env/bin/python")
    
    try:
        # Run the test script and capture output
        result = subprocess.run(
            [python_path, test_script],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract NDI sources from the output
        sources = []
        lines = result.stdout.splitlines()
        capture = False
        
        for line in lines:
            if "Found NDI sources" in line:
                capture = True
                continue
            if capture and line.strip().startswith("Source"):
                # Extract the source name
                parts = line.split(":", 1)
                if len(parts) > 1:
                    sources.append(parts[1].strip())
            if capture and line.strip() == "":
                # End of sources list
                capture = False
        
        return sources
    except Exception as e:
        print(f"Error getting NDI sources: {e}")
        return []

def run_ndi_example(args):
    """Run the NDI example with the specified parameters."""
    # Get the path to Python in the ndi-env
    python_path = os.path.expanduser("~/.pyenv/versions/3.10.12/envs/ndi-env/bin/python")
    main_script = os.path.join(repo_root, "examples/ndi/main.py")
    
    # Build the command
    cmd = [python_path, main_script]
    
    # Add parameters
    if args.model:
        cmd.extend(["--model_id_or_path", args.model])
    if args.prompt:
        cmd.extend(["--prompt", args.prompt])
    if args.negative_prompt:
        cmd.extend(["--negative_prompt", args.negative_prompt])
    if args.width:
        cmd.extend(["--width", str(args.width)])
    if args.height:
        cmd.extend(["--height", str(args.height)])
    if args.source:
        cmd.extend(["--ndi_source_name", args.source])
    if args.output:
        cmd.extend(["--ndi_output_name", args.output])
    
    # Print the command
    print("Running command:")
    print(" ".join(cmd))
    
    # Run the command
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping NDI example...")
    except Exception as e:
        print(f"Error running NDI example: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Run the StreamDiffusion NDI example with Python 3.10 NDI bindings"
    )
    
    # List available NDI sources
    sources = get_ndi_sources()
    
    parser.add_argument(
        "--model",
        type=str,
        default="KBlueLeaf/kohaku-v2.1",
        help="Model ID or path to use for diffusion"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="1girl with brown dog hair, thick glasses, smiling",
        help="Prompt to guide the diffusion"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, bad quality, blurry, low resolution",
        help="Negative prompt to avoid certain characteristics"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Width of the output image"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of the output image"
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=sources if sources else None,
        help="NDI source to use (will use first available if not specified)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="StreamDiffusion Output",
        help="Name for the NDI output stream"
    )
    
    # Print available sources
    if sources:
        print("Available NDI sources:")
        for i, source in enumerate(sources):
            print(f"  {i+1}. {source}")
    else:
        print("No NDI sources found")
    
    args = parser.parse_args()
    run_ndi_example(args)

if __name__ == "__main__":
    main()