#!/bin/bash

#this script is for setting up myelocare project on windows using wsl2 - cuda - pytorch


# >> prerequisites

    #install wsl through followin commands
    #wsl --install -d Ubuntu
    #after installing wsl, open ubuntu and run the following commands (or this script)
    #make sure to install the latest nvidia drivers for windows from nvidia website




# windows terminal commands

# =========================================================

#wsl --install -d Ubuntu

#code -install-extension ms-vscode-remote.remote-wsl


# ==========================================================
# Update system
sudo apt update -y
sudo apt upgrade -y

# Install dependencies
sudo apt install -y gcc g++ build-essential python3 python3-pip python3-venv curl tar

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