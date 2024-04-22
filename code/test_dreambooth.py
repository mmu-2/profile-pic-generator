from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch

import glob


model_id = "./dreambooth-model"
# prompt = "a photo of mu person"
# prompt = "mu* person posing in the woods"
# prompt = "mu* at the beach"
# prompt = "mu* facing me from a skyscraper"
prompt = "mu* standing in front of the mona lisa"
# prompt = "mu* on a mountain"


pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                            torch_dtype=torch.float16).to("cuda")
pipe.safety_checker = None
pipe.requires_safety_checker = False
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image.save("mu.png")


for checkpoint in glob.glob('./dreambooth-model/checkpoint-*/'):
    iteration = checkpoint.split('-')[-1][:-1] #want to ignore last /

    unet = UNet2DConditionModel.from_pretrained(f"./dreambooth-model/checkpoint-{iteration}/unet", torch_dtype=torch.float16).to("cuda")

    pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                unet=unet,
                                                torch_dtype=torch.float16).to("cuda")
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    image.save(f"mu-{iteration}.png")


