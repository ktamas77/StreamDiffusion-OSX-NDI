# NDI/Webcam Stream Processing with StreamDiffusion

This example demonstrates how to:
1. Receive video from an NDI source or webcam
2. Apply StreamDiffusion with a text prompt
3. Display the processed result on screen
4. Broadcast the processed video as an NDI output (if NDI is available)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## NDI Support

This example supports NDI if the appropriate SDK is installed:

1. Download and install the [NDI SDK](https://ndi.tv/sdk/) for your platform
2. Install a Python NDI binding (many options exist - the code will try to import common ones)

If NDI is not available, the example will automatically fall back to using your webcam as input.

## Usage

Run the example:

```bash
python main.py --prompt "your prompt here" 
```

### Options

- `--ndi_source_name`: Name of the NDI source to connect to (will use first available if not specified)
- `--ndi_output_name`: Name for the NDI output stream (default: "StreamDiffusion Output")
- `--prompt`: Text prompt to guide the diffusion process
- `--negative_prompt`: Negative prompt to avoid certain characteristics
- `--width`, `--height`: Dimensions for processing (default: 512x512)
- `--model_id_or_path`: Model ID or path to use for diffusion

For all options, run:

```bash
python main.py --help
```

## Requirements

- macOS with compatible hardware
- StreamDiffusion-OSX
- OpenCV (for webcam fallback)
- Optional: NDI SDK and Python binding

## Notes

- For optimal performance, ensure you have a fast GPU or Apple Silicon device