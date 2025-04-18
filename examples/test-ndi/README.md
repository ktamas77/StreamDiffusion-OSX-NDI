# NDI Testing and Integration for StreamDiffusion on macOS

This directory contains tools to test and work with NDI (Network Device Interface) on macOS for StreamDiffusion.

## Contents

- `test_ndi.py` - Diagnostic tool to check if NDI SDK is properly installed and accessible
- `fallback_ndi.py` - A fallback NDI module that simulates NDI functionality using webcam input when the actual NDI SDK is not available
- `ndi_driver.py` - A unified driver interface for NDI that works with both real NDI bindings and the fallback simulator
- `setup_ndi_env.sh` - Script to set up a Python environment with NDI bindings
- `requirements.txt` - Complete list of Python dependencies for the NDI-enabled environment

## NDI SDK Installation Status

The test script shows that the NDI library is installed on your system at `/usr/local/lib/libndi.dylib`, but the Python bindings need to be installed separately.

## Installation Instructions

### Install Python 3.10.12 with pyenv

NDI Python bindings are compatible with Python 3.7-3.10, but not with Python 3.13. We'll use Python 3.10.12.

```bash
# Install pyenv and pyenv-virtualenv if you don't have them
brew install pyenv pyenv-virtualenv

# Add pyenv to your shell configuration (if not already done)
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
source ~/.zshrc

# Install Python 3.10.12
pyenv install 3.10.12

# Create a virtual environment
pyenv virtualenv 3.10.12 ndi-env
```

### Set up NDI Python environment

```bash
# Clone the repository if you haven't already
git clone https://github.com/ktamas77/StreamDiffusion-OSX.git
cd StreamDiffusion-OSX

# Activate the environment
pyenv activate ndi-env

# Install NDI Python bindings
pip install ndi-python

# Install StreamDiffusion dependencies
pip install -e .
pip install diffusers transformers accelerate torchvision
```

Alternatively, you can use the automatic setup script:

```bash
cd StreamDiffusion-OSX
./examples/test-ndi/setup_ndi_env.sh
```

Or install from the requirements.txt:

```bash
pyenv activate ndi-env
pip install -r examples/test-ndi/requirements.txt
pip install -e .
```

### Test NDI Installation

```bash
cd StreamDiffusion-OSX

# Using pyenv
pyenv activate ndi-env
python test_ndi_installation.py

# Or using the explicit Python path
~/.pyenv/versions/3.10.12/envs/ndi-env/bin/python test_ndi_installation.py
```

## Running StreamDiffusion with NDI

```bash
cd StreamDiffusion-OSX

# Activate the environment
pyenv activate ndi-env

# Run the NDI example
python examples/ndi/main.py
```

If you have problems with the activation command, you can use the explicit path:

```bash
~/.pyenv/versions/3.10.12/envs/ndi-env/bin/python examples/ndi/main.py
```

## Using the NDI Driver

The `ndi_driver.py` script provides a unified interface for NDI functionality whether using the real NDI bindings or the fallback simulator. Here's how to use it in your own code:

```python
from examples.test-ndi.ndi_driver import NDIHandler

# Create an NDI handler
ndi = NDIHandler()

# Find NDI sources
sources = ndi.find_sources()
if sources:
    # Create a receiver for the first source
    source = sources[0]
    receiver_id = ndi.create_receiver(source)
    
    # Create a sender
    sender_id = ndi.create_sender("My NDI Output")
    
    # Receive and send frames
    for i in range(100):  # Process 100 frames
        frame = ndi.receive_frame(receiver_id)
        if frame:
            ndi.send_frame(sender_id, frame)

# Clean up
ndi.cleanup()
```

## Troubleshooting

### NDI Sources Not Found

If no NDI sources are found when running the test, this is normal if there are no NDI sources broadcasting on your network. You can:

1. Install NDI Studio Monitor and NDI Virtual Input from the NDI Tools package: https://ndi.tv/tools/
2. Run NDI Virtual Input to create an NDI source from your webcam

### Python Version Issues

The ndi-python package is compatible with Python 3.7-3.10 but not with Python 3.13. Make sure you're using the correct Python version.

### Shell Activation Issues

If you have problems with `pyenv activate ndi-env`, you can:

1. Restart your terminal to ensure pyenv initialization is loaded
2. Use the explicit Python path: `~/.pyenv/versions/3.10.12/envs/ndi-env/bin/python`

### Webcam Access

The fallback NDI module tries to use your webcam, but macOS requires permission. Check System Settings > Privacy & Security > Camera to ensure Terminal has permission.

## Additional Resources

- NDI SDK: https://ndi.tv/sdk/
- NDI Tools: https://ndi.tv/tools/
- ndi-python GitHub: https://github.com/buresu/ndi-python