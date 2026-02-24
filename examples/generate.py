#!/usr/bin/env python3
from util import post_request, encode_image_to_base64

MODEL_PATH = 'wangqixun/YamerMIX_v8'
#MODEL_PATH = '/workspace/runpod-worker-instantid/src/models/dynavisionXLAllInOneStylized_releaseV0610Bakedvae.safetensors'
IMAGE_PATH = '../data/mypic.jpg'
PROMPT = 'a man'
NEGATIVE_PROMPT = 'nsfw'
STYLE = 'Watercolor'
#STYLE = 'Retro Futurism'
NUM_STEPS = 30
IDENTITYNET_STRENGTH_RATIO = 0.8
ADAPTER_STRENGTH_RATIO = 0.8
GUIDANCE_SCALE = 5
SEED = 42


if __name__ == '__main__':
    payload = {
        "input": {
            "model": MODEL_PATH,
            "face_image": encode_image_to_base64(IMAGE_PATH),
            "pose_image": None,
            "prompt":  PROMPT,
            "negative_prompt": NEGATIVE_PROMPT,
            "style_name": STYLE,
            "num_steps": NUM_STEPS,
            "identitynet_strength_ratio": IDENTITYNET_STRENGTH_RATIO,
            "adapter_strength_ratio":  ADAPTER_STRENGTH_RATIO,
            "guidance_scale": GUIDANCE_SCALE,
            "seed": SEED,
            # "width": 640,
            # "height": 860
        }
    }

    post_request(payload)
