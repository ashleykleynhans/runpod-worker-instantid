#!/usr/bin/env bash

echo "Worker Initiated"

echo "Starting RunPod Handler"
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=1
cd /workspace/runpod-worker-instantid/src
python3 -u handler.py
