#!/bin/bash
# Script to set up an NDI-compatible Python environment and install NDI Python bindings

# Define colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up NDI Python Environment${NC}"
echo "----------------------------------------"

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo -e "${RED}Error: pyenv is not installed${NC}"
    echo "Please install pyenv first:"
    echo "    brew install pyenv pyenv-virtualenv"
    exit 1
fi

# Check if the environment variable is set up
if ! grep -q "pyenv init" ~/.zshrc; then
    echo -e "${YELLOW}Adding pyenv to your shell configuration...${NC}"
    echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
    echo 'eval "$(pyenv init -)"' >> ~/.zshrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
    echo -e "${YELLOW}Please restart your terminal or run:${NC}"
    echo '    source ~/.zshrc'
fi

# Install Python 3.10.12 if not already installed
if ! pyenv versions | grep -q "3.10.12"; then
    echo -e "${GREEN}Installing Python 3.10.12...${NC}"
    pyenv install 3.10.12
else
    echo -e "${GREEN}Python 3.10.12 is already installed${NC}"
fi

# Create a virtual environment if it doesn't exist
if ! pyenv versions | grep -q "ndi-env"; then
    echo -e "${GREEN}Creating virtual environment ndi-env...${NC}"
    pyenv virtualenv 3.10.12 ndi-env
else
    echo -e "${GREEN}Virtual environment ndi-env already exists${NC}"
fi

# Activate the environment and install packages
echo -e "${GREEN}Activating ndi-env and installing packages...${NC}"
# Set the local Python version for this directory
pyenv local ndi-env

# Install packages - using eval to run within the current script context
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
pyenv activate ndi-env

# Install/upgrade packages
pip install --upgrade pip wheel setuptools
pip install ndi-python

# Create a test script
cat > ./test_ndi_installation.py << EOL
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
EOL

chmod +x ./test_ndi_installation.py

echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo -e "${YELLOW}To use this environment in a new terminal, run:${NC}"
echo "    cd $(pwd)"
echo "    pyenv activate ndi-env"
echo ""
echo -e "${YELLOW}To test the NDI installation, run:${NC}"
echo "    ./test_ndi_installation.py"
echo ""
echo -e "${YELLOW}Now running the test script:${NC}"
python ./test_ndi_installation.py