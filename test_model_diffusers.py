import argparse
import accelerate
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import os 
from glob import glob 
from tqdm import tqdm
import torch
from PIL import Image 

import diffusers
from diffusers import (
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
)
from pipeline_stable_diffusion_controlnet import StableDiffusionControlNetPipeline
from controlnet_customed import ControlNetModel
from torchvision import transforms

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models--runwayml--stable-diffusion-v1-5",
        #required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default="../transformers_controlnet_cache",
        #required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--style_data_root",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument('--use_vae_encoder_for_style', 
                        action='store_true', 
                        default=False,
                        help='whether using vae encoder to extract style, not used')
    parser.add_argument(
        "--unet_feature_fusion_type",
        type=str,
        default="addition",
        help="how to fuse the controlnet output and the feature of unet, only addition supported",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--num_inference_step", type=int, default=25, help="number of inference steps.")
    parser.add_argument("--cfg", type=float, default=10.0, help="classifer-free guidance.")
    parser.add_argument("--controlnet_scale", type=float, default=0.7, help="controlnet scale.")
    
    args = parser.parse_args()
    return args

def setup_pipeline(args):
    weight_dtype = torch.float32
    
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, 
                                                 cache_dir=args.model_cache_dir, 
                                                 local_files_only=True)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            controlnet=controlnet,
            revision=args.revision,
            torch_dtype=weight_dtype,
            local_files_only=True,
            variant="fp16",
        )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
    pipeline.safety_checker = None
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=pipeline.device).manual_seed(args.seed)

    return pipeline, generator, None

def gen_single_image(
                    args,
                    prompt, 
                    cond_image_path, 
                    pipeline, 
                    generator=None, 
                    vae=None,
                    ):
    weight_dtype = torch.float32
    
    control_image = Image.open(cond_image_path).convert("RGB")
    control_image = control_image.resize((512, 512), resample=Image.Resampling.BILINEAR)
    
    if args.use_vae_encoder_for_style:
        image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        control_image = image_transforms(control_image).to("cuda").unsqueeze(0)
        control_image = vae.encode(control_image.to(dtype=weight_dtype)).latent_dist.sample()
        control_image = control_image * vae.config.scaling_factor
    
    image =  pipeline(
                        prompt, 
                        control_image, 
                        num_inference_steps=args.num_inference_step, 
                        generator=generator,
                        height=512,
                        width=512,
                        guidance_scale=args.cfg,
                        controlnet_conditioning_scale=args.controlnet_scale,
                        #unet_fusion_type=unet_feature_fusion_type,
                ).images[0]
    
    return image

def main():
    args = parse_args()

    pipeline, generator, _ = setup_pipeline(args)
    #vae = vae.to("cuda")

    style_images = glob(os.path.join(args.style_data_root, "*"))
    prompt = args.prompt + ", best quality"

    sub_folder = args.save_dir
    os.makedirs(sub_folder, exist_ok=True)
    
    print(len(style_images))
    
    for cond_image in tqdm(style_images):
        print(cond_image)
        image_name = os.path.basename(cond_image)
        try:
            im = Image.open(cond_image)
        except:
            continue
        image = gen_single_image(
                                    args,
                                    prompt, 
                                    cond_image, 
                                    pipeline, 
                                    generator,
                                    #vae,
                                ).convert("RGB")
        image.save(os.path.join(sub_folder, image_name))

if __name__ == "__main__":
    main()
    