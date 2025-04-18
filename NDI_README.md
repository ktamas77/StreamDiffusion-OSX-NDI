# NDI Integration for StreamDiffusion

This document provides comprehensive information about the NDI (Network Device Interface) integration with StreamDiffusion on macOS.

## What is NDI?

NDI (Network Device Interface) is a high-performance standard for video-over-IP that allows software applications to send and receive high-quality, low-latency video and audio streams over standard Ethernet networks.

## Features

- **NDI Input**: Capture video from any NDI source (cameras, software outputs, etc.)
- **NDI Output**: Output processed video as an NDI source for use in other applications
- **OSC Control**: Real-time control of AI parameters via Open Sound Control (OSC)
- **Fallback Mode**: Works with webcam if NDI sources are not available
- **Cross-application Integration**: Compatible with OBS, Resolume, TouchDesigner, etc.

## Setup

### Prerequisites

- macOS (tested on macOS Ventura and newer)
- Python 3.10+ (Python 3.10.12 recommended for best compatibility)
- [NDI SDK](https://ndi.video/download-ndi-sdk/) (optional - fallback mode works without it)

### Installation

1. Set up a virtual environment:
   ```bash
   python -m venv ndi_venv
   source ndi_venv/bin/activate
   ```

2. Install required packages:
   ```bash
   pip install -r examples/ndi-osc/requirements.txt
   ```

3. Set up NDI Python bindings (optional):
   ```bash
   python examples/test-ndi/setup_ndi.py
   ```

4. Test NDI functionality:
   ```bash
   python examples/test-ndi/test_ndi.py
   ```

## Available Implementations

### 1. Basic NDI Implementation

Located in `examples/ndi/run_fixed_ndi.py`, this provides basic NDI input/output functionality.

Usage:
```bash
python examples/ndi/run_fixed_ndi.py --ndi_source_name "Your NDI Source"
```

### 2. NDI with OSC Control

Located in `examples/ndi-osc/run_ndi_osc.py`, this adds OSC control for real-time prompt changes.

Usage:
```bash
python examples/ndi-osc/run_ndi_osc.py --ndi_source_name "Your NDI Source" --osc_port 9001
```

Available OSC commands:
- `/prompt <text>` - Set a new prompt for image generation
- `/negative_prompt <text>` - Set a new negative prompt
- `/guidance_scale <float>` - Set a new guidance scale value (typically 1.0-7.0)

## NDI Python Bindings

The repository includes the `ndi-python` directory with Python bindings for NDI. Due to compatibility issues with newer Python versions, these bindings may not compile on all systems. In such cases, the implementation automatically falls back to using a webcam-based simulator.

To check your NDI setup:
```bash
python examples/test-ndi/test_ndi.py
```

## Troubleshooting

### NDI Sources Not Found

If you have NDI sources running but they're not detected:
1. Check if NDI SDK is properly installed
2. Verify that NDI sources are on the same network
3. Check for any firewall restrictions
4. Restart the NDI services

### Python Version Issues

The NDI Python bindings work best with Python 3.10. If you experience issues:
1. Install Python 3.10 using pyenv or your preferred method
2. Create a new virtual environment with Python 3.10
3. Follow the installation steps above

### Build Errors

If you encounter build errors with the NDI Python bindings:
1. Make sure you have CMake and a C++ compiler installed
2. Check that the NDI SDK is properly installed
3. Set the necessary environment variables as indicated by the test script

If problems persist, the implementation will automatically fall back to webcam mode.

## Integration Examples

### OBS Studio

1. Start an NDI output from this application
2. In OBS, add an NDI Source and select the output

### TouchDesigner

1. Use an NDI IN TOP to receive the processed output
2. Use an OSC OUT CHOP to send control messages to the application

### Resolume

1. Add an NDI source in the Sources panel
2. Select the StreamDiffusion output from the list

## Advanced Configuration

See the individual script files for additional command-line arguments and configuration options.