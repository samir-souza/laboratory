# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
os.environ["NEURON_FUSE_SOFTMAX"] = "1"
import time
import copy
import torch
import shutil
import argparse
import numpy as np
import torch_neuronx
import torch.nn as nn

from wrapper import NeuronTextEncoder, UNetWrap, NeuronUNet, get_attention_scores
from diffusers.models.cross_attention import CrossAttention
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def compile_text_encoder(text_encoder, args):
    print("Compiling text encoder...")
    base_dir='text_encoder'
    os.makedirs(os.path.join(args.checkpoints_path, base_dir), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, base_dir), exist_ok=True)
    t = time.time()
    # Apply the wrapper to deal with custom return type
    text_encoder = NeuronTextEncoder(text_encoder)

    # Compile text encoder
    # This is used for indexing a lookup table in torch.nn.Embedding,
    # so using random numbers may give errors (out of range).
    emb = torch.tensor([[49406, 18376,   525,  7496, 49407,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0]])
    text_encoder_neuron = torch_neuronx.trace(
        text_encoder.neuron_text_encoder, emb,
        #compiler_workdir=os.path.join(args.checkpoints_path, base_dir),
    )

    # Save the compiled text encoder
    text_encoder_filename = os.path.join(args.model_path, base_dir, 'model.pt')
    torch.jit.save(text_encoder_neuron, text_encoder_filename)

    # delete unused objects
    del text_encoder
    del text_encoder_neuron
    print(f"Done. Elapsed time: {(time.time()-t)*1000}ms")

def compile_vae(decoder, args, dtype):
    print("Compiling VAE...")
    base_dir='vae_decoder'
    os.makedirs(os.path.join(args.checkpoints_path, base_dir), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, base_dir), exist_ok=True)
    t = time.time()
    # Compile vae decoder
    decoder_in = torch.randn([1, 4, 64, 64]).type(dtype)
    decoder_neuron = torch_neuronx.trace(
        decoder,
        decoder_in,
        #compiler_workdir=os.path.join(args.checkpoints_path, base_dir),
    )

    # Save the compiled vae decoder
    decoder_filename = os.path.join(args.model_path, base_dir, 'model.pt')
    torch.jit.save(decoder_neuron, decoder_filename)

    # delete unused objects
    del decoder
    del decoder_neuron
    print(f"Done. Elapsed time: {(time.time()-t)*1000}ms")

def compile_unet(unet, args, dtype):
    print("Compiling U-Net...")
    base_dir='unet'
    os.makedirs(os.path.join(args.checkpoints_path, base_dir), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, base_dir), exist_ok=True)
    t = time.time()
    # Compile unet - BF16
    sample_1b = torch.randn([1, 4, 64, 64]).type(dtype)
    timestep_1b = torch.tensor(999).type(dtype).expand((1,))
    encoder_hidden_states_1b = torch.randn([1, 77, 1024]).type(dtype)
    example_inputs = sample_1b, timestep_1b, encoder_hidden_states_1b

    unet_neuron = torch_neuronx.trace(
        unet,
        example_inputs,
        #compiler_workdir=os.path.join(args.checkpoints_path, base_dir),
        compiler_args=["--model-type=unet-inference"]
    )

    # save compiled unet
    unet_filename = os.path.join(args.model_path, base_dir, 'model.pt')
    torch.jit.save(unet_neuron, unet_filename)

    # delete unused objects
    del unet
    del unet_neuron
    print(f"Done. Elapsed time: {(time.time()-t)*1000}ms")

def compile_vae_post_quant_conv(post_quant_conv, args, dtype):
    print("Compiling Post Quant Conv...")
    base_dir='vae_post_quant_conv'
    os.makedirs(os.path.join(args.checkpoints_path, base_dir), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, base_dir), exist_ok=True)
    t = time.time()

    # # Compile vae post_quant_conv
    post_quant_conv_in = torch.randn([1, 4, 64, 64]).type(dtype)
    post_quant_conv_neuron = torch_neuronx.trace(
        post_quant_conv,
        post_quant_conv_in,
        #compiler_workdir=os.path.join(args.checkpoints_path, base_dir),
    )

    # # Save the compiled vae post_quant_conv
    post_quant_conv_filename = os.path.join(args.model_path, base_dir, 'model.pt')
    torch.jit.save(post_quant_conv_neuron, post_quant_conv_filename)

    # delete unused objects
    del post_quant_conv
    del post_quant_conv_neuron
    print(f"Done. Elapsed time: {(time.time()-t)*1000}ms")

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model-path', type=str, help="Path where we'll save the model", default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--checkpoints-path', type=str, help="Path where we'll save the best model and cache", default='/opt/ml/checkpoints')
    parser.add_argument('--dtype', type=str, choices=['bf16','fp32'], default='bf16', help="Datatype of the weights")

    args = parser.parse_args()

    # make sure the checkpoint path exists
    os.makedirs(args.checkpoints_path, exist_ok=True)

    # Model ID for SD version pipeline
    model_id = "stabilityai/stable-diffusion-2-1-base"

    # --- Compile CLIP text encoder and save ---

    dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32
    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    text_encoder = copy.deepcopy(pipe.text_encoder)
    del pipe
    compile_text_encoder(text_encoder, args)

    # --- Compile VAE decoder and save ---

    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    decoder = copy.deepcopy(pipe.vae.decoder)
    del pipe
    compile_vae(decoder, args, dtype)

    # --- Compile UNet and save ---

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # Replace original cross-attention module with custom cross-attention module for better performance
    CrossAttention.get_attention_scores = get_attention_scores

    # Apply double wrapper to deal with custom return type
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))

    # Only keep the model being compiled in RAM to minimze memory pressure
    unet = copy.deepcopy(pipe.unet.unetwrap)
    del pipe
    compile_unet(unet, args, dtype)

    # --- Compile VAE post_quant_conv and save ---

    # Only keep the model being compiled in RAM to minimze memory pressure
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
    post_quant_conv = copy.deepcopy(pipe.vae.post_quant_conv)
    del pipe
    compile_vae_post_quant_conv(post_quant_conv, args, dtype)

    code_path = os.path.join(args.model_path, 'code')
    os.makedirs(code_path, exist_ok=True)
    
    # replace datatype
    with open('inference.py', 'r') as f:
        data = f.read()
    data = data.replace('DTYPE_REPLACE', 'torch.bfloat16' if args.dtype=='bf16' else 'torch.float32')
    with open('inference.py', 'w') as f:
        f.write(data)

    shutil.copyfile('inference.py', os.path.join(code_path, 'inference.py'))
    shutil.copyfile('wrapper.py', os.path.join(code_path, 'wrapper.py'))
    shutil.copyfile('requirements.txt', os.path.join(code_path, 'requirements.txt'))