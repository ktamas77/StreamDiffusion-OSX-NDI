# StreamDiffusion-OSX

StreamDiffusion implementation optimized for macOS. This fork extends the original StreamDiffusion with macOS compatibility and adds enhanced NDI (Network Device Interface) support.

## Overview

This repository contains a macOS-compatible version of StreamDiffusion, allowing real-time image generation and processing on Apple Silicon and Intel Macs. With added NDI support, you can use StreamDiffusion with video applications like OBS, Zoom, and other NDI-compatible software.

### Key Features

- **macOS Compatibility**: Optimized for Apple Silicon and Intel Macs
- **NDI Support**: Stream to and from NDI sources
- **OSC Control**: Real-time prompt control via OSC messages
- **Real-time Image Generation**: Process video streams in real-time with AI
- **Multiple Examples**: Various sample applications showing different use cases

## NDI Support

This fork adds comprehensive NDI support, allowing StreamDiffusion to:

1. **Receive video from NDI sources**: Connect to any NDI source on your network
2. **Process video with AI**: Apply AI-generated styles and effects
3. **Broadcast processed video**: Output the result as an NDI source

### NDI Example (`examples/ndi/`)

The NDI example demonstrates how to use StreamDiffusion with NDI sources and outputs. See the [NDI Example README](examples/ndi/README.md) for detailed instructions.

```bash
# Run the NDI example (shows available NDI sources)
./examples/ndi/run_fixed_ndi.py

# With specific NDI source and prompt
./examples/ndi/run_fixed_ndi.py --ndi_source_name "STUDIO (Camera)" \
                                --prompt "cyberpunk neon city, detailed"
```

### NDI-OSC Integration (`examples/ndi-osc/`)

The NDI-OSC example adds real-time creative control via OSC messages. This allows you to change prompts and parameters on the fly using tools like TouchOSC, Max/MSP, or any OSC-capable software. See the [NDI-OSC README](examples/ndi-osc/README.md) for details.

```bash
# Run the NDI-OSC server
./examples/ndi-osc/run_ndi_osc.py

# Send OSC commands from another application or terminal
# Examples (using oscsend or similar tools):
oscsend localhost 9001 /prompt s "surreal landscape, dreamy atmosphere"
oscsend localhost 9001 /guidance_scale f 2.5
```

### NDI Testing Tools (`examples/test-ndi/`)

A suite of tools for testing NDI functionality and diagnosing issues. See the [NDI Testing README](examples/test-ndi/README.md) for details.

```bash
# Test NDI installation
./examples/test-ndi/test_ndi.py

# Set up NDI environment
./examples/test-ndi/setup_ndi_env.sh
```

## Requirements

- macOS 11 or later
- Python 3.10 recommended for NDI support (Python 3.13 not compatible with NDI)
- NDI SDK installed (optional, falls back to webcam if not available)
- python-osc (for NDI-OSC integration)

## Installation

For detailed installation instructions, see the [Installation Guide](examples/test-ndi/INSTALL.md).

```bash
# Clone this repository
git clone https://github.com/ktamas77/StreamDiffusion-OSX-NDI.git
cd StreamDiffusion-OSX-NDI

# Set up Python environment (for NDI support)
pyenv install 3.10.12
pyenv virtualenv 3.10.12 ndi-env
pyenv activate ndi-env

# Basic installation
pip install -e .
pip install -r examples/ndi/requirements.txt

# For OSC support
pip install python-osc==1.8.3
```

## Creative Applications

This fork enables several creative applications:

- **VJ Systems**: Connect to VJ software for AI-enhanced visuals
- **Live Performance**: Control the AI parameters in real-time via OSC
- **Video Production**: Process NDI sources from cameras or software
- **Interactive Installations**: Create responsive AI visual experiences

## Original Documentation

For the original StreamDiffusion documentation and examples, see the [Original README](README_original.md).

## License

This project is licensed under the same license as the original StreamDiffusion project.