from huggingface_hub import snapshot_download


# Everything here is symlinked to /home/ec2-user/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/


# https://huggingface.co/runwayml/stable-diffusion-v1-5
# model = "runwayml/stable-diffusion-v1-5"

# https://huggingface.co/OFA-Sys/small-stable-diffusion-v0/tree/main
# model = "OFA-Sys/small-stable-diffusion-v0"

# https://huggingface.co/segmind/portrait-finetuned/tree/main?library=true
model = "segmind/portrait-finetuned"


snapshot_download(repo_id=model,
                local_dir="./",
                local_dir_use_symlinks=True)