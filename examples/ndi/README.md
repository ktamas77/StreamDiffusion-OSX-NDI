# NDI Example for StreamDiffusion on macOS

This example demonstrates using NDI (Network Device Interface) with StreamDiffusion for real-time video processing. It can receive video from NDI sources, apply AI-based image generation, and output the processed video as an NDI stream.

## Requirements

- macOS (tested on macOS 15.3.2)
- Python 3.10 (NDI Python bindings do not work with Python 3.13)
- NDI SDK installed (see instructions in `examples/test-ndi/INSTALL.md`)
- StreamDiffusion dependencies

## Setup

Follow the setup instructions in `examples/test-ndi/INSTALL.md` to set up a Python 3.10 environment with NDI bindings.

## Quick Start

The easiest way to run this example is using the provided `run_ndi.py` script:

```bash
# Navigate to the repository root
cd /path/to/StreamDiffusion-OSX

# Run the helper script (it will show available NDI sources)
./examples/ndi/run_ndi.py

# Run with a specific source
./examples/ndi/run_ndi.py --source "MACSTUDIOABLETON.LOCAL (macOS AV Output)"

# Run with a custom prompt
./examples/ndi/run_ndi.py --prompt "a magical forest with glowing trees, fantasy art style"
```

## Manual Execution

If you prefer to run the script directly:

```bash
# Activate the ndi-env environment (after following setup instructions)
pyenv activate ndi-env  # or use the absolute path if shell activation doesn't work

# Run the main script
python examples/ndi/main.py

# With custom options
python examples/ndi/main.py --ndi_source_name "MACSTUDIOABLETON.LOCAL (macOS AV Output)" \
                             --prompt "a magical forest with glowing trees, fantasy art style" \
                             --width 512 --height 512
```

## Command Line Arguments

The main script accepts the following parameters:

- `--model_id_or_path`: Model ID or path (default: "KBlueLeaf/kohaku-v2.1")
- `--prompt`: Prompt for image generation
- `--negative_prompt`: Negative prompt for image generation
- `--width`: Output width (default: 512)
- `--height`: Output height (default: 512)
- `--ndi_source_name`: Name of NDI source to use (default: first available)
- `--ndi_output_name`: Name for the NDI output stream (default: "StreamDiffusion Output")

## How It Works

1. The script connects to an NDI source (or falls back to webcam if none available)
2. Video frames are processed through the StreamDiffusion pipeline
3. The processed frames are:
   - Displayed in a viewer window
   - Sent to an NDI output stream
4. Other applications can receive the output NDI stream

## Troubleshooting

- **NDI sources not detected**: Make sure you have NDI sources active on your network. You can use the NDI Studio Monitor app (part of the NDI Tools package) to check for available sources.
- **Python activation issues**: Use the absolute path to Python in the ndi-env if shell activation fails: `~/.pyenv/versions/3.10.12/envs/ndi-env/bin/python`
- **Model download issues**: The first run may take some time as it downloads the model. Ensure you have a good internet connection.
- **Performance issues**: Try reducing the width and height values for better performance on lower-end hardware.