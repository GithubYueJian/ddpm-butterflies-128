
from cfg import TrainingConfig
from diffusers import DDPMScheduler, UNet2DModel
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
from diffusers.utils import make_image_grid

config = TrainingConfig()
scheduler = DDPMScheduler.from_pretrained("./ddpm-butterflies-128",subfolder="scheduler")
# print(type(scheduler.timesteps))
# print(scheduler.timesteps)
# sys.exit()
model = UNet2DModel.from_pretrained("./ddpm-butterflies-128", subfolder="unet",use_safetensors=True).to("cuda")


batch_size = 16
sample_size = config.image_size
noise = torch.randn((batch_size, 3, sample_size, sample_size), device="cuda")

input = noise


for t in tqdm(scheduler.timesteps,desc="inference"):
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample


images = (input / 2 + 0.5).clamp(0, 1).squeeze()
images = (images.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
for i in tqdm(range(batch_size),desc="saving"):
    image = Image.fromarray(images[i])
    image.save(f"./test_dir/test_butterfly_{i}.png")