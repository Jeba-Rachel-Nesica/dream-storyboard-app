from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

def stylize_img2img(input_img, style_preset, device='cpu'):
    # Use SD img2img with style prompt
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    prompt = style_preset
    generator = torch.Generator(device).manual_seed(42)
    img = pipe(prompt=prompt, image=input_img, strength=0.7, guidance_scale=7.5, generator=generator).images[0]
    return img
