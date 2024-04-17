import argparse
import safetensors
import torch


def save_progress(text_encoder, placeholder_token_ids, args, save_path, safe_serialization=True):
    print("Saving embeddings")
    learned_embeds = (
        text_encoder.get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save learned_embeds.bin every X updates steps.",)
    parser.add_argument("--save_as_full_pipeline",action="store_true",help="Save the complete stable diffusion pipeline.",)
    parser.add_argument("--num_vectors",type=int,default=1,help="How many textual inversion vectors shall be used to learn the concept.",)
    parser.add_argument("--pretrained_model_name_or_path",type=str,default=None,required=True,help="Path to pretrained model or model identifier from huggingface.co/models.",)
    parser.add_argument("--revision",type=str,default=None,required=False,help="Revision of pretrained model identifier from huggingface.co/models.",)
    parser.add_argument("--variant",type=str,default=None,help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",)
    parser.add_argument("--tokenizer_name",type=str,default=None,help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data.")
    parser.add_argument("--placeholder_token",type=str,default=None,required=True,help="A token to use as a placeholder for the concept.",)
    parser.add_argument("--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word.")
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument("--output_dir",type=str,default="text-inversion-model",help="The output directory where the model predictions and checkpoints will be written.",)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution",type=int,default=512,help=("The resolution for input images, all the images in the train/validation dataset will be resized to this resolution"),)
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps",type=int,default=5000,help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing",action="store_true",help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)
    parser.add_argument("--learning_rate",type=float,default=1e-4,help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--lr_scheduler",type=str,default="constant",help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'),)
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles",type=int,default=1,help="Number of hard resets of the lr in cosine_with_restarts scheduler.",)
    parser.add_argument("--dataloader_num_workers",type=int,default=0,help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."),)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--mixed_precision",type=str,default="no",choices=["no", "fp16", "bf16"],help=("Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."),)
    parser.add_argument("--allow_tf32",action="store_true",help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),)
    parser.add_argument("--report_to",type=str,default="tensorboard",help=('The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'),)
    parser.add_argument("--validation_prompt",type=str,default=None,help="A prompt that is used during validation to verify that the model is learning.",)
    parser.add_argument("--num_validation_images",type=int,default=4,help="Number of images that should be generated during validation with `validation_prompt`.",)
    parser.add_argument("--validation_steps",type=int,default=100,help=("Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images."),)
    parser.add_argument("--checkpointing_steps",type=int,default=500,help=("Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`."),)
    parser.add_argument("--checkpoints_total_limit",type=int,default=None,help=("Max number of checkpoints to store."),)
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,help=("Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or \"latest\" to automatically select the last available checkpoint."),)
    parser.add_argument("--no_safe_serialization",action="store_true",help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",)
    args = parser.parse_args()

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")
    return args