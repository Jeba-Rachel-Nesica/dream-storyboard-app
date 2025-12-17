import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
import random
from app.pipelines.identity import compute_face_similarity

# Provider interface: can swap to ComfyUI/Ollama if needed
class DiffusionProvider:
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        print(f"[DiffusionProvider] Using device: {self.device}")
        self.pipe = None
        self.model_path = model_path or "runwayml/stable-diffusion-v1-5"
    
    def _lazy_load(self):
        if self.pipe is None:
            self.pipe = StableDiffusionPipeline.from_pretrained(self.model_path).to(self.device)

    def generate(self, prompt, negative_prompt, seed, ref_img=None, face_emb=None):
        self._lazy_load()
        generator = torch.Generator(self.device).manual_seed(seed)
        if ref_img is not None:
            img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_path).to(self.device)
            img = img_pipe(prompt=prompt, negative_prompt=negative_prompt, image=ref_img, strength=0.7, guidance_scale=7.5, generator=generator).images[0]
        else:
            img = self.pipe(prompt=prompt, negative_prompt=negative_prompt, guidance_scale=7.5, generator=generator).images[0]
        return img

provider = DiffusionProvider()

def generate_candidates(beat, identity, n=3):
    imgs = []
    for i in range(n):
        seed = random.randint(0, 2**32-1)
        img = provider.generate(
            prompt=beat['positive_prompt'],
            negative_prompt=beat['negative_prompt'],
            seed=seed,
            ref_img=identity['best_crop'],
            face_emb=identity['agg_embedding']
        )
        imgs.append((img, {'seed': seed}))
    return imgs

def auto_rank_candidates(candidates, identity):
    # Optionally rank by face similarity
    ranked = []
    for img, meta in candidates:
        sim = compute_face_similarity(img, identity['agg_embedding'])
        meta['sim'] = sim
        ranked.append((img, meta))
    ranked.sort(key=lambda x: -x[1]['sim'])
    return ranked
