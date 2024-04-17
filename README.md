# profile-pic-generator


## Work notes

Base textual inversion will take 1 hour on a V100 on https://huggingface.co/docs/diffusers/en/training/text_inversion.

Not sure whether to do:
- pip install torch torchvision torchaudio
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

I tried to use https://huggingface.co/runwayml/stable-diffusion-v1-5, but I found that the model was too big and I get OOM errors
no matter how many quantization tricks I used.

I switched to https://huggingface.co/OFA-Sys/small-stable-diffusion-v0

Weirdly enough, batch_size is changing runtime by a lot:
- 4 - 1.1 hours
- 16 - 5 hours
- 24 - 7.5 hours.


1. Setup environment. See requirements.txt for packages.
2. In models, download the file you need in download.py, else you need to modify filepath in training.
3. In code, run train.

## Run

python train.py  --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/cat_toy_example" --placeholder_token  "<mu*>" --initializer_token "toy" --learnable_property "object" --mixed_precision fp16 --enable_xformers_memory_efficient_attention

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/cat_toy_example" --placeholder_token  "<mu*>" --initializer_token "toy" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4