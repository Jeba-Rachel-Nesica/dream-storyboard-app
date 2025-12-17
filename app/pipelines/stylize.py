from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

def stylize_img2img(input_img, style_preset, device='cpu', strength=0.5):
    # Use SD img2img with style prompt
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    prompt = style_preset
    generator = torch.Generator(device).manual_seed(42)
    # Lower strength (0.5 instead of 0.7) for better identity preservation
    img = pipe(prompt=prompt, image=input_img, strength=strength, guidance_scale=7.5, generator=generator).images[0]
    return img
