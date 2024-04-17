from diffusers import StableDiffusionPipeline
import torch

# model_id = "text-inversion-model"
model_id = "../models/"
pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
pipe.load_textual_inversion("./text-inversion-model/", weight_name="learned_embeds.safetensors",)

prompt = "<mu*> on top of a desk"

image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("me.png")