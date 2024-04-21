from diffusers import StableDiffusionPipeline
import torch
import glob

# model_id = "text-inversion-model"

def test_textual_inversion():
    for name in glob.glob('/home/ec2-user/profile-pic-generator/code/text-inversion-model/*.safetensors'): 
        filename = name.split('/')[-1]
        iterations = filename.split('-')[-1].split('.')[0]
        model_id = "../models/"
        pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
        pipe.load_textual_inversion("./text-inversion-model/", weight_name=filename)

        # For humans, sometimes it will trigger a NSFW filter, but I want to see them.
        # pipe.safety_checker = lambda images, clip_input: (images, False)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False

        # prompt = "<mu*> standing in front of a white wall"
        prompt = "a photo of one <mu*>"
    #     prompt = """Faceshot Portrait of a young (24-year-old) <mu*> wearing a green sweater, 
    # (masterpiece, extremely detailed skin, photorealistic, heavy shadow, dramatic and cinematic 
    # lighting, key light, fill light), sharp focus, BREAK epicrealism"""

        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

        image.save(f"me-{iterations}.png")
        del pipe

def test_base():
    model_id = "../models/"
    pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
    # pipe.load_textual_inversion("./text-inversion-model/", weight_name=filename)

    # prompt = "a chinese-american man in a green sweater and tan pants standing in front of a white wall"
    prompt = """Faceshot Portrait of a young (24-year-old) chinese american wearing a green sweater, 
    (masterpiece, extremely detailed skin, photorealistic, heavy shadow, dramatic and cinematic 
    lighting, key light, fill light), sharp focus, BREAK epicrealism"""

    negative = """(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, 
    sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, 
    worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, 
    extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, 
    deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, 
    disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, 
    extra legs, fused fingers, too many fingers, long neck"""

    image = pipe(prompt, negative_prompt=negative, num_inference_steps=50, guidance_scale=7.5).images[0]

    image.save(f"before-training.png")

if __name__ == "__main__":

    test_textual_inversion()
    test_base()