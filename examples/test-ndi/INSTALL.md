# Step-by-Step Installation Guide for NDI with StreamDiffusion

This guide provides detailed instructions for setting up NDI with StreamDiffusion on macOS.

## Prerequisites

- macOS (tested on 15.3.2)
- Homebrew package manager
- NDI SDK installed (library found at `/usr/local/lib/libndi.dylib`)
- Git

## Step 1: Install pyenv and pyenv-virtualenv

pyenv allows you to easily switch between multiple versions of Python.

```bash
# Install pyenv and pyenv-virtualenv
brew install pyenv pyenv-virtualenv

# Set up pyenv in your shell configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc

# Apply changes
source ~/.zshrc
```

## Step 2: Install Python 3.10.12

Python 3.10.12 is required for compatibility with NDI Python bindings.

```bash
# Install Python 3.10.12
pyenv install 3.10.12

# Verify the installation
pyenv versions
```

## Step 3: Create and Activate a Virtual Environment

```bash
# Create a virtual environment named 'ndi-env'
pyenv virtualenv 3.10.12 ndi-env

# Activate the environment
pyenv activate ndi-env

# Verify the Python version
python --version  # Should show Python 3.10.12
```

## Step 4: Clone the StreamDiffusion Repository

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/ktamas77/StreamDiffusion-OSX.git
cd StreamDiffusion-OSX
```

## Step 5: Install NDI Python Bindings and Dependencies

```bash
# Install core dependencies with correct versions
pip install --upgrade pip wheel setuptools
pip install ndi-python==5.1.1.5
pip install -r examples/test-ndi/requirements.txt

# Install StreamDiffusion in development mode
pip install -e .

# Install additional dependencies for StreamDiffusion
pip install torch==2.6.0 torchvision==0.21.0
pip install diffusers==0.33.1 transformers==4.51.3 accelerate==1.6.0
```

## Step 6: Test the NDI Installation

```bash
# Run the test script
python test_ndi_installation.py
```

You should see output that includes:
- "NDIlib is correctly installed!"
- "NDI successfully initialized"
- A list of available NDI sources (if any are broadcasting on your network)

## Step 7: Run StreamDiffusion with NDI

```bash
# Run the NDI example
python examples/ndi/main.py
```

## Troubleshooting

### Terminal shows "Failed to activate virtualenv"

If you see this error:
```
Failed to activate virtualenv.
Perhaps pyenv-virtualenv has not been loaded into your shell properly.
Please restart current shell and try again.
```

Solution:
1. Restart your terminal
2. Try using the absolute path to Python:
   ```bash
   ~/.pyenv/versions/3.10.12/envs/ndi-env/bin/python examples/ndi/main.py
   ```

### No NDI Sources Found

If no NDI sources are detected when running the test script:

1. Install NDI Tools from https://ndi.tv/tools/
2. Run NDI Virtual Input to create an NDI source from your webcam
3. Run NDI Studio Monitor to view NDI sources on your network

### Error: "module 'NDIlib' has no attribute..."

Different versions of NDI Python bindings may have slightly different APIs. The `ndi_driver.py` module provides a compatibility layer for these differences.

## Installing NDI SDK (If Not Already Installed)

If the NDI SDK is not installed (test shows no library at `/usr/local/lib/libndi.dylib`):

1. Download the NDI SDK from https://ndi.tv/sdk/ (requires registration)
2. Follow the installation instructions for macOS
3. Verify installation by running `ls -la /usr/local/lib/libndi*`