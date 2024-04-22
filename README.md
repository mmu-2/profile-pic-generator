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

If using base aws linux, nothing will be installed and you should do the following:
1. sudo yum install python -y
2. sudo yum install pip -y
3. sudo yum install git -y
4. git clone https://github.com/mmu-2/profile-pic-generator.git
5. pip install virtualenv
6. cd ~
6. virtualenv pytorch
7. source ~/pytorch/bin/activate

https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#nvidia-gaming-driver
8. sudo yum install gcc make -y
9. sudo yum update -y
10. sudo reboot
11. sudo yum install -y gcc kernel-devel-$(uname -r)
12. 1. Get key from: https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html
12. 2. aws configure list
12. 3. region: us-east-2
12. 4. default output format: [Enter]
12. aws s3 cp --recursive s3://nvidia-gaming/linux/latest/ .
13. unzip latest-driver-name.zip -d nvidia-drivers
14. chmod +x nvidia-drivers/550.73-Apr2024-Cloud_Gaming-Linux-Guest-Drivers/NVIDIA-Linux-x86_64*-grid.run
15. sudo ./nvidia-drivers/550.73-Apr2024-Cloud_Gaming-Linux-Guest-Drivers/NVIDIA-Linux-x86_64*.run

1. Setup environment. See requirements.txt for packages.
2. In models, download the file you need in download.py, else you need to modify filepath in training.
3. In code, run train.

## Run

python train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/cat_toy_example" --placeholder_token "<mu*>" --initializer_token "toy" --learnable_property "object" --mixed_precision fp16 --enable_xformers_memory_efficient_attention

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/cat_toy_example" --placeholder_token "<mu*>" --initializer_token "toy" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 8

## Experiment 1

I got the baseline working with example cat toy example. Now, I want to see how it works with me.
Current experiment: Number of training iterations

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/me" --placeholder_token "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 8

python test.py

Result: Really bad looking humans

## Experiment 2

The output of the previous experiment is worse than just a descriptive prompt. This tells me that something has gone severely wrong.
To remedy this, I did several things: crop the images to focus more on me and reduce the number of duplicated training data samples for training.

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/me" --placeholder_token "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50

python test.py


## Experiment 3
4/19/24a
The experiments are not turning out too well for human outputs, so I am switching models to https://huggingface.co/segmind/portrait-finetuned

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/me" --placeholder_token "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50

python test.py

Results: There are more humans in the outputs, but they still look demonic.

I also switched test.py to have more specific prompts. The results are better, but the textual inversions isn't.

## Experiment 4
more specific prompts and detailed prompts and try multiple vectors

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/me" --placeholder_token "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50 --num_vectors 2

python test.py

Result: No improvement on 3.

## Experiment 5
More vectors

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/me" --placeholder_token "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50 --num_vectors 8

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

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/my_portrait" --placeholder_token "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50 --num_vectors 8

python test.py

Result: Somewhat improved.

## Experiment 9

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/my_portrait" --placeholder_token "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 50 --num_vectors 8

python test.py

Result: Some improvement.

## Experiment 10
4 data samples, reduced repeats.

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/my_portrait" --placeholder_token "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 4 --repeats 25 --num_vectors 8

python test.py

## Experiment 11
4 data samples, reduced repeats.

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/my_portrait" --placeholder_token "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 8 --repeats 25 --num_vectors 16

python test.py


## Experiment 12
Batch size 2

accelerate launch train.py --pretrained_model_name_or_path "../models/" --train_data_dir "../datasets/my_portrait" --placeholder_token "<mu*>" --initializer_token "person" --learnable_property "object" --mixed_precision "fp16" --enable_xformers_memory_efficient_attention --train_batch_size 2 --repeats 25 --num_vectors 16

python test.py

## Experiment 13
Switch to dreambooth. Note this does not do prior preservation, which should be okay for this use case.

This one definitely works

accelerate launch train_dreambooth.py --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" --instance_data_dir "../datasets/my_portrait" --output_dir="dreambooth-model" --instance_prompt="a photo of mu person" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=800 --mixed_precision "fp16" --use_8bit_adam

python test_dreambooth.py

## Experiment 14
Newer model.

accelerate launch train_dreambooth.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" --instance_data_dir "../datasets/my_portrait" --output_dir="dreambooth-model" --instance_prompt="a photo of mu person" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=1600 --mixed_precision "fp16" --use_8bit_adam

python test_dreambooth.py

Results: Seems a little worse than 1-4 honestly.

## Experiment 15

accelerate launch train_dreambooth.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" --instance_data_dir "../datasets/me" --output_dir="dreambooth-model" --instance_prompt="a photo of mu* smiling and standing in a green sweater" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=3000 --mixed_precision "fp16" --use_8bit_adam --checkpointing_steps 1000

python test_dreambooth.py

## Experiment 16

accelerate launch train_dreambooth.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" --instance_data_dir "../datasets/me" --output_dir="dreambooth-model" --instance_prompt="a photo of mu* smiling and standing in a green sweater" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=6000 --mixed_precision "fp16" --use_8bit_adam --checkpointing_steps 2000

python test_dreambooth.py

Results: Not so good.

## Experiment 17

accelerate launch train_dreambooth.py --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" --instance_data_dir "../datasets/mix" --output_dir="dreambooth-model" --instance_prompt="a photo of mu* smiling and standing" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=1 --learning_rate=5e-6 --lr_scheduler="constant" --lr_warmup_steps=0 --max_train_steps=6000 --mixed_precision "fp16" --use_8bit_adam --checkpointing_steps 2000

python test_dreambooth.py
