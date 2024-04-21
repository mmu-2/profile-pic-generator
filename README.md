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
- 8 - 2.5 hours
- 16 - 5 hours
- 24 - 7.5 hours.


1. Setup environment. See requirements.txt for packages.
2. In models, download the file you need in download.py, else you need to modify filepath in training.
3. In code, run train.

## Run

python train.py  --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/cat_toy_example" --placeholder_token  "<mu*>" --initializer_token "toy" --learnable_property "object" --mixed_precision fp16 --enable_xformers_memory_efficient_attention

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/cat_toy_example" --placeholder_token  "<mu*>" --initializer_token "toy" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 8

## Experiment 1

I got the baseline working with example cat toy example. Now, I want to see how it works with me.
Current experiment: Number of training iterations

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/me" --placeholder_token  "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 8

python test.py

Result: Really bad looking humans

## Experiment 2

The output of the previous experiment is worse than just a descriptive prompt. This tells me that something has gone severely wrong.
To remedy this, I did several things: crop the images to focus more on me and reduce the number of duplicated training data samples for training.

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/me" --placeholder_token  "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50

python test.py


## Experiment 3
4/19/24a
The experiments are not turning out too well for human outputs, so I am switching models to https://huggingface.co/segmind/portrait-finetuned

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/me" --placeholder_token  "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50

python test.py

Results: There are more humans in the outputs, but they still look demonic.

I also switched test.py to have more specific prompts. The results are better, but the textual inversions isn't.

## Experiment 4
more specific prompts and detailed prompts and try multiple vectors

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/me" --placeholder_token  "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50 --num_vectors 2

python test.py

Result: No improvement on 3.

## Experiment 5
More vectors

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/me" --placeholder_token  "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50 --num_vectors 8

python test.py

Result: Slightly better? Can't really tell.

## Experiment 6
Try 1000 iterations for timesteps.

python test.py

Result: All the images turned black, it must not have been trained for this many steps.

## Experiment 7
Back to 50 iterations for timesteps. Turn guidance down to 7.5 from 15.

python test.py

Result: No real change.

## Experiment 8
Change prompt to "<mu*> standing in front of a white wall"

python test.py

Result: No real change.

## Experiment 9
Cropped to just my face.

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/my_portrait" --placeholder_token  "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50 --num_vectors 8

python test.py

Result: Somewhat improved.

## Experiment 9

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/my_portrait" --placeholder_token  "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50 --num_vectors 8

python test.py

Result: Some improvement.

## Experiment 10
4 data samples, reduced repeats.

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/my_portrait" --placeholder_token  "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 25 --num_vectors 8

python test.py

## Experiment 11
4 data samples, reduced repeats.

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/my_portrait" --placeholder_token  "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 8 --repeats 25 --num_vectors 16

python test.py


## Experiment 12
Batch size 2

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/my_portrait" --placeholder_token  "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 2 --repeats 25 --num_vectors 16

python test.py

## Experiment 13
Switch to dreambooth. Note this does not do prior preservation, which should be okay for this use case.

accelerate launch train_dreambooth.py --pretrained_model_name_or_path "../models/" --instance_data_dir "../datasets/my_portrait" --output_dir="dreambooth-model" --instance_prompt="a photo of mu person" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=2 --gradient_checkpointing --use_8bit_adam --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=800 --mixed_precision "fp16"

accelerate launch train_dreambooth.py --pretrained_model_name_or_path "../models/" --instance_data_dir "../datasets/my_portrait" --output_dir="dreambooth-model" --instance_prompt="a photo of mu person" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=800 --mixed_precision "fp16"

accelerate launch train_dreambooth.py --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" --instance_data_dir "../datasets/my_portrait" --output_dir="dreambooth-model" --instance_prompt="a photo of mu person" --resolution=512 --train_batch_size=1


export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="../datasets/my_portrait"
export CLASS_DIR="../datasets/class"
export OUTPUT_DIR="dreambooth-model"

  accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800