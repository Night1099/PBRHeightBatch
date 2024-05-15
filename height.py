import argparse
import random
import os

import argparse
import random
import os
from PIL import Image, ImageOps, ImageFilter
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler	
from diffusers.utils import load_image
import torch
from diffusers.pipelines.stable_diffusion import safety_checker

# bypass nsfw false positives
def sc(self, clip_input, images) : return images, [False for i in images]
safety_checker.StableDiffusionSafetyChecker.forward = sc

parser = argparse.ArgumentParser(description="Args for parser")
parser.add_argument("--seed", type=int, default=random.randint(0, 100000), help="Seed for inference")
parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output images")
parser.add_argument("--append", type=str, default='', help="String to append to the end of original file's basename")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_path = "runwayml/stable-diffusion-v1-5"
controlnet_path = "NightRaven109/ControlnetHeightPBR"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.bfloat16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.bfloat16
)

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15_light_v11.bin")
pipe.set_ip_adapter_scale(0.5)

pipe.scheduler = 	EulerAncestralDiscreteScheduler	.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "height map texture of brick wall"  # Positive prompt
negative_prompt = "patterns"

generator = torch.manual_seed(args.seed)

for filename in os.listdir(args.input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        control_image = load_image(os.path.join(args.input_dir, filename))

        ip_image = control_image
        max_size = (1024, 1024)
        if control_image.size[0] > max_size[0] or control_image.size[1] > max_size[1]:
            control_image.thumbnail(max_size, Image.LANCZOS)

        image = pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=5,
            generator=generator,
            image=control_image,
            ip_adapter_image=ip_image,
            controlnet_conditioning_scale=0.7,
            guidance_scale=4,
            scheduler=pipe.scheduler,
        ).images[0]

        image = ImageOps.grayscale(image)
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
        output_filename = os.path.splitext(filename)[0] + args.append + '.png'
        image.save(os.path.join(args.output_dir, output_filename))

