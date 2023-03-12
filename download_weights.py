#!/usr/bin/env python

import os
import sys
import torch
import shutil
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

from transformers import CLIPTextModel, CLIPTokenizer
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

def load_model(MODEL_ID, MODEL_CACHE, VAE_ID, VAE_CACHE):
    MODEL_PATH = f"{MODEL_CACHE}/{MODEL_ID}"
    VAE_PATH = f"{VAE_CACHE}/{VAE_ID}"

    if not os.path.exists(MODEL_PATH):
        print("Downloading Base Training Model: " + MODEL_ID)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
        )
        os.makedirs(MODEL_PATH, exist_ok=True)
        pipe.save_pretrained(MODEL_PATH)
    else:
        print("Loading Base Training Model Cache: " + MODEL_ID)
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_PATH,
        )
    
    pipe.enable_xformers_memory_efficient_attention()

    if not os.path.exists(VAE_PATH):
        print("Downloading VAE: " + VAE_ID)
        pretrained_vae = AutoencoderKL.from_pretrained(
            VAE_ID,
        )
        pretrained_vae.save_pretrained(VAE_PATH)
        os.makedirs(VAE_PATH, exist_ok=True)
    else:
        print("Loading VAE Cache: " + VAE_ID)
        pretrained_vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
        )

    pretrained_vae.enable_xformers_memory_efficient_attention()