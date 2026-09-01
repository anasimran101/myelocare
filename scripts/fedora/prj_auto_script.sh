#!/bin/bash

#this script is for setting up myelocare project on windows using wsl2 - cuda - pytorch


# >> prerequisites

    #install wsl through followin commands
    #wsl --install -d Ubuntu
    #after installing wsl, open ubuntu and run the following commands (or this script)
    #make sure to install the latest nvidia drivers for windows from nvidia website

# NOTE: This script is adapted for Fedora WSL
# To install Fedora on WSL: wsl --install -d Fedora
# Or download from: https://github.com/WhitewaterFoundry/Fedora-Remix-for-WSL




# windows terminal commands

# =========================================================

#wsl --install -d Fedora

#code -install-extension ms-vscode-remote.remote-wsl


# ==========================================================
# Update system
sudo dnf update -y
sudo dnf upgrade -y

# Install dependencies
# Note: @development-tools group includes gcc, g++, make, etc.
sudo dnf install -y @development-tools python3 python3-pip python3-virtualenv curl tar

# Install Python venv module (if not already installed)
sudo dnf install -y python3-venv

# Create project folder
git clone https://github.com/anasimran101/myelocare.git

# Create virtual environment
python3 -m venv myelocare_env
source myelocare_env/bin/activate

# Install PyTorch with CUDA support (no Linux driver needed)
pip install torch torchvision

# Install Ultralytics (YOLO)
pip install ultralytics

# Install LazyGit
LAZYGIT_VERSION=$(curl -s https://api.github.com/repos/jesseduffield/lazygit/releases/latest | grep tag_name | cut -d '"' -f 4)
curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/download/${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION#v}_Linux_x86_64.tar.gz"
tar xf lazygit.tar.gz lazygit
sudo install lazygit /usr/local/bin
rm -f lazygit lazygit.tar.gz

# Test PyTorch and Ultralytics
python3 -c "import torch; import ultralytics; print('CUDA available:', torch.cuda.is_available()); print('Ultralytics version:', ultralytics.__version__)"