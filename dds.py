# %%
import wandb
import sd_utils
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from PIL import Image
import numpy as np


# %%
img_name = 'flamingo.png'
prompt_ref = "a flamingo rollerskating"
prompt = "a stork rollerskating"
guidance_scale = 15
batch_size = 1

wandb.init(project="dds")
guidance_model = sd_utils.StableDiffusion('cuda', fp16=True, vram_O=False)
guidance_model.eval()
for p in guidance_model.parameters():
    p.requires_grad = False

def load_image_as_tensor(image_path):
    # Load the image with PIL
    img = Image.open(image_path).convert("RGB")

    # Convert the image to a PyTorch tensor
    img_tensor = torch.from_numpy(np.array(img)).half().permute(2, 0, 1)

    # Normalize pixel values to range [0, 1]
    img_tensor /= 255

    # Add batch dimension
    batch_img_tensor = img_tensor.unsqueeze(0)

    return batch_img_tensor

img_ref = load_image_as_tensor(img_name).cuda()
img_ref.requires_grad = False

as_latent = True
latent_size = 96
latent = torch.randn(1, 4, latent_size, latent_size, requires_grad=True, device="cuda")

with torch.no_grad():
    text_z_ref = torch.cat([guidance_model.get_text_embeds(""), guidance_model.get_text_embeds(prompt_ref), ], dim=0)
    text_z = torch.cat([guidance_model.get_text_embeds(""), guidance_model.get_text_embeds(prompt), ], dim=0)
    latent_ref = guidance_model.encode_imgs(img_ref)
    latent[:] = latent_ref

# optim = torch.optim.Adam([latent], lr=0.01)
optim = torch.optim.SGD([latent], lr=0.02)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 20, 0.9)
# %%
import torch.nn.functional as F

cos_sim = []

for i in tqdm(range(401)):
    optim.zero_grad()
    x = latent.half()

    noise_pred, t, noise = guidance_model.predict_noise(text_z, x, guidance_scale=guidance_scale, as_latent=as_latent)
    with torch.no_grad():
        noise_pred_ref, _, _ = guidance_model.predict_noise(text_z_ref, latent_ref, guidance_scale=guidance_scale, as_latent=as_latent, t=t, noise=noise)
    
    if i % 20 == 0:
        with torch.no_grad():
            wandb.log({"cos sim": F.cosine_similarity(noise_pred - noise, noise_pred_ref - noise).mean().item()})
            if as_latent:
                wandb.log({"result": wandb.Image(guidance_model.decode_latents(x)[0])})
            else:
                plt.imshow(latent[0].detach().cpu().permute(1, 2, 0))
        plt.show()
    w =  (1 - guidance_model.alphas[t])
    grad = w * (noise_pred - noise_pred_ref)
    grad = torch.nan_to_num(grad)

    loss = sd_utils.SpecifyGradient.apply(x, grad)
    loss.backward()
    optim.step()
    scheduler.step()

# %%