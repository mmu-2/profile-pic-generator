# profile-pic-generator


## Work notes

Base textual inversion will take 1 hour on a V100 on https://huggingface.co/docs/diffusers/en/training/text_inversion.

Not sure whether to do:
- pip install torch torchvision torchaudio
- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

## Run

python train.py  --pretrained_model_name_or_path "asdf" --train_data_dir "../datasets/cat_toy_example" --placeholder_token  "<mu*>" --initializer_token "toy" --learnable_property "object"