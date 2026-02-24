#!/usr/bin/env bash

echo "Worker Initiated"

if [[ ! -L /workspace ]]; then
  echo "Symlinking files from Network Volume"
  ln -s /runpod-volume /workspace
fi

if [ -f "/workspace/venv/bin/python3" ]; then
    echo "Starting RunPod Handler"
    TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
    export LD_PRELOAD="${TCMALLOC}"
    export PYTHONUNBUFFERED=1
    export HUGGINGFACE_HUB_CACHE="/runpod-volume/huggingface-cache/hub"
    export TRANSFORMERS_CACHE="/runpod-volume/huggingface-cache/hub"
    cd /workspace/runpod-worker-instantid/src
    /workspace/venv/bin/python3 -u handler.py
else
    echo "ERROR: The Python Virtual Environment (/workspace/venv) could not be found"
    echo "1. Ensure that you have followed the instructions at: https://github.com/ashleykleynhans/runpod-worker-instantid/blob/main/docs/building/with-network-volume.md"
    echo "2. Ensure that you have used the Pytorch image for the installation and NOT an InstantID image."
    echo "3. Ensure that you have attached your Network Volume to your endpoint."
    echo "4. Ensure that you didn't assign any other invalid regions to your endpoint."
fi
