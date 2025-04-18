# NDI-OSC Integration for StreamDiffusion

This example demonstrates using NDI (Network Device Interface) with StreamDiffusion for real-time video processing, with the added capability of controlling prompt parameters via OSC (Open Sound Control) messages.

## Features

- **NDI Input**: Connect to any NDI source on your network
- **OSC Control**: Real-time prompt updating via OSC messages
- **NDI Output**: Output the processed video as an NDI source
- **Visual Feedback**: OpenCV window shows the processed output

## How It Works

1. **Video Input**: The script connects to an NDI source (or falls back to webcam)
2. **AI Processing**: StreamDiffusion applies AI-generated styles based on the prompt
3. **OSC Server**: Listens for OSC messages to update parameters in real-time
4. **Rendering**: Processed frames are displayed and sent as NDI output
5. **Dynamic Control**: Prompts and parameters can be changed at any time via OSC

## Requirements

- macOS (tested on macOS 15.3.2)
- Python 3.10 (NDI Python bindings do not work with Python 3.13)
- NDI SDK installed (see instructions in `examples/test-ndi/INSTALL.md`)
- StreamDiffusion dependencies
- Python-OSC library

## Setup

Follow the setup instructions in `examples/test-ndi/INSTALL.md` to set up a Python 3.10 environment with NDI bindings, then:

```bash
# Activate the environment
pyenv activate ndi-env

# Install additional dependencies
pip install python-osc==1.8.3

# If you have activation issues, use the absolute path:
~/.pyenv/versions/3.10.12/envs/ndi-env/bin/pip install python-osc==1.8.3
```

## Usage

```bash
# Run with default settings (with proper Python environment)
python examples/ndi-osc/run_ndi_osc.py

# If you have activation issues, use the absolute path:
~/.pyenv/versions/3.10.12/envs/ndi-env/bin/python examples/ndi-osc/run_ndi_osc.py

# Run with specific NDI source
python examples/ndi-osc/run_ndi_osc.py --ndi_source_name "STUDIO (Camera)"

# Run with custom OSC port
python examples/ndi-osc/run_ndi_osc.py --osc_port 8000

# Run with custom width/height for better performance
python examples/ndi-osc/run_ndi_osc.py --width 256 --height 256
```

## OSC Commands

Send OSC messages to the specified IP and port (default: 127.0.0.1:9001) with these commands:

| OSC Address | Parameters | Description |
|-------------|------------|-------------|
| `/prompt` | String | Set new prompt for image generation |
| `/negative_prompt` | String | Set negative prompt |
| `/guidance_scale` | Float | Set guidance scale (typically 1.0-7.0) |

### Example OSC Messages

Using [oscsend](https://github.com/yoggy/sendosc) or similar tools:

```bash
# Set a new prompt
oscsend localhost 9001 /prompt s "surreal landscape, dreamy atmosphere, Magritte style"

# Set a negative prompt
oscsend localhost 9001 /negative_prompt s "blurry, bad quality, low resolution"

# Set guidance scale
oscsend localhost 9001 /guidance_scale f 2.5
```

Using Max/MSP, TouchOSC, or other OSC applications, send messages to the proper address with appropriate types.

## Command Line Arguments

The script accepts the following parameters:

- `--model_id_or_path`: Model ID or path (default: "KBlueLeaf/kohaku-v2.1")
- `--prompt`: Initial prompt (default: "cyberpunk neon city, vaporwave aesthetic, vibrant colors")
- `--negative_prompt`: Initial negative prompt
- `--width`: Output width (default: 512)
- `--height`: Output height (default: 512)
- `--ndi_source_name`: Name of NDI source to use
- `--ndi_output_name`: Name for the NDI output stream (default: "StreamDiffusion Output")
- `--osc_ip`: IP address for OSC server (default: "127.0.0.1")
- `--osc_port`: Port for OSC server (default: 9001)
- `--guidance_scale`: Initial guidance scale (default: 1.4)
- `--delta`: Delta for noise scheduling (default: 0.5)
- `--acceleration`: Acceleration method ("none", "xformers", "tensorrt")
- `--seed`: Random seed (default: 2)

## Creative Applications

This integration enables several creative use cases:

- **Live VJ Performance**: Change visual styles in real-time to match music
- **Interactive Installations**: Let users control the AI through interfaces that send OSC
- **Experimental Filmmaking**: Create unique visual effects controlled via OSC
- **Mixed Media Performance**: Integrate with DAWs and performance software

## Troubleshooting

- **OSC messages not received**: Check firewall settings and that the correct IP and port are used
- **NDI sources not detected**: Make sure NDI sources are active on your network
- **Python errors**: Ensure you're using Python 3.10 with all dependencies installed
- **Performance issues**: Try reducing width and height for better performance
- **Model loading issues**: First run may take time to download the model

## Integrations

This setup works well with:
- **Max/MSP** and Pure Data for creative applications
- **TouchOSC** for mobile control
- **VJ software** like Resolume or VDMX
- **DAWs** like Ableton Live with OSC capabilities
- **Custom interfaces** built with OSC-capable frameworks

## For Developers

If you want to extend this implementation:

- Add new OSC commands in the handler functions
- The OSC server runs in a separate thread to avoid blocking the main process
- All parameters are synchronized using thread locks to prevent race conditions