#!/usr/bin/env bash

TORCH_VERSION="2.6.0"
XFORMERS_VERSION="0.0.29.post3"

echo "Deleting InstantID Serverless Worker"
rm -rf /workspace/runpod-worker-instantid

echo "Deleting venv"
rm -rf /workspace/venv

echo "Cloning InstantID Serverless Worker repo to /workspace"
cd /workspace
git clone https://github.com/ashleykleynhans/runpod-worker-instantid.git
cd runpod-worker-instantid

echo "Installing Ubuntu updates"
apt update
apt -y upgrade

echo "Installing git-lfs"
apt -y install git-lfs
git lfs install

echo "Creating and activating venv"
cd /workspace/runpod-worker-instantid
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

echo "Installing Torch"
pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing xformers"
pip3 install --no-cache-dir xformers==${XFORMERS_VERSION}

echo "Installing RunPod SDK"
pip3 install --no-cache-dir runpod

echo "Installing InstantID Serverless Worker"
pip3 install -r src/requirements.txt

echo "Installing checkpoints"
cd /workspace/runpod-worker-instantid/src
export HUGGINGFACE_HUB_CACHE="/workspace/huggingface-cache/hub"
export TRANSFORMERS_CACHE="/workspace/huggingface-cache/hub"
python3 download_checkpoints.py

echo "Downloading antelopev2 models from Huggingface"
git clone https://huggingface.co/ashleykleynhans/FaceAnalysis models

echo "Creating log directory"
mkdir -p /workspace/logs
