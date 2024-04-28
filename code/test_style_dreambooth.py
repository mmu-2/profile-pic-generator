import PIL.Image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionImg2ImgPipeline
# from custom_img2img import StableDiffusionImg2ImgPipeline
import torch
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp

import torchvision.transforms as transforms

import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default=None, required=True, help="directory of saved model")
parser.add_argument("--src_img", type=str, default=None, required=True, help="image to perform style transfer on")
parser.add_argument("--strength", type=float, default=0.8, required=False, help=
                    """Amount of noise to add for SDEdit. Must be between 0 and 1. 1 is like no image is added. 
                    0 is just keeping the image""")
parser.add_argument("--prompt", type=str, default=None, required=True, help="prompt for style transfer on")
args = parser.parse_args()

model_id = args.model_dir
imagefile = args.src_img
strength = args.strength

img = PIL.Image.open(imagefile)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(512),
])
img = transform(img)#.permute(0,2,1)
# img = transform(img).permute(0,2,1)


pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,
                                                      torch_dtype=torch.float16).to("cuda")
pipe.safety_checker = None
pipe.requires_safety_checker = False
# pipe.enable_sequential_cpu_offload()
# pipe.enable_attention_slicing()
# pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
# pipe.enable_vae_tiling()
img = pipe(args.prompt, img, strength=strength, num_inference_steps=50, guidance_scale=7.5).images[0]
img.save(f"stylized.png")

for checkpoint in glob.glob('./dreambooth-model/checkpoint-*/'):
    iteration = checkpoint.split('-')[-1][:-1] #want to ignore last /

    unet = UNet2DConditionModel.from_pretrained(f"./dreambooth-model/checkpoint-{iteration}/unet", torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,
                                                unet=unet,
                                                torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    output = pipe(args.prompt, img, strength=strength, num_inference_steps=50, guidance_scale=7.5)
    img = output.images[0]
    img.save(f"stylized-{iteration}.png")


