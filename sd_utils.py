from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from dataclasses import dataclass


class SpecifyGradient(torch.autograd.Function):
    """
    This code defines a custom gradient function using PyTorch's `torch.autograd.Function` class. It is particularly helpful when you want to manipulate gradients manually in a deep learning model that relies on automatic differentiation. The class is called `SpecifyGradient`, and contains two essential methods: `forward` and `backward`.

1. The `@staticmethod` decorator indicates that these are static methods and can be called on the class itself, without instantiating an object from the class.

2. The `forward` method takes two input arguments: `ctx` and `input_tensor`. `ctx` is a context object used to store information needed for backward computation. `input_tensor` is the input tensor to this layer in the neural network. The purpose of this method is to compute the forward pass and store any required information for the backward pass.

3. The `@custom_fwd` decorator is a user-defined decorator (not provided here) which presumably wraps or modifies the forward method in some way, most likely to add functionality like logging, error checking or other custom behavior.

4. Inside the `forward` method, the ground truth gradient `gt_grad` is saved using `ctx.save_for_backward()`. This stored information will be used later in the backward function. The forward function then returns a tensor of ones with the same device and data type as the input tensor. This tensor will be used in the backward pass as a scaling factor to adjust the gradients.

5. The `backward` method takes two input arguments: `ctx` and `grad_scale`. `ctx` is the same context object used in the forward pass. `grad_scale` is the gradient scaling factor used to adjust the gradients. The purpose of this method is to compute the gradient updates with respect to the input during backpropagation. 

6. The `@custom_bwd` decorator is another user-defined decorator (not provided here) which performs a similar role for the backward method as the `@custom_fwd` decorator does for the forward method.

7. Inside the `backward` method, the ground truth gradient `gt_grad` is retrieved from the saved tensors. It is then scaled by multiplying it with `grad_scale`. The method returns the scaled gradient `gt_grad` and `None`. The `None` value is returned because there are no gradients to compute for `gt_grad` with respect to the input tensor – it is assumed to be an external property that doesn't require gradient computation.

This custom gradient function can be used in situations where you need to have fine-grained control over the gradients in a neural network. For example, if you want to perform gradient clipping or apply noise to the gradients, you would use this `SpecifyGradient` function in place of a standard PyTorch layer.
    """
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

@dataclass
class UNet2DConditionOutput:
    sample: torch.HalfTensor # Not sure how to check what unet_traced.pt contains, and user wants. HalfTensor or FloatTensor

class StableDiffusion(nn.Module):
    """
    This is a Python class `StableDiffusion` that inherits from `nn.Module` in PyTorch. 
    `StableDiffusion` is a trainable generative model based on the Stable Diffusion framework, 
    which is an image-conditional generative model that can produce high fidelity images from text prompts. 
    The class has various methods, including `__init__`, `get_text_embeds`, `train_step`, `produce_latents`, `decode_latents`, and `prompt_to_img`. 
    """
    def __init__(self, device, fp16, vram_O, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98]):
        """
The `__init__` method initializes the class and loads a Stable Diffusion model using the specified version number, 
and also sets the precision of the model to either float16 or float32 depending on the `fp16` parameter. 
It sets the device the model will run on based on the `device` parameter. 
If a `hf_key` parameter is provided, it will use the Hugging Face custom model key specified, 
otherwise it will use a pre-trained Stable Diffusion model based on the `sd_version` parameter. 
It also initializes a `StableDiffusionPipeline` object that contains various sub-models such as `text_encoder`, `vae`, and `unet`, 
as well as a `DDIMScheduler` object that controls the number of training steps. 
        """
        super().__init__()

        self.device = device
        self.sd_version = sd_version

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.precision_t)

        if isfile('./unet_traced.pt'):
            # use jitted unet
            unet_traced = torch.jit.load('./unet_traced.pt')
            class TracedUNet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.in_channels = pipe.unet.in_channels
                    self.device = pipe.unet.device

                def forward(self, latent_model_input, t, encoder_hidden_states):
                    sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
                    return UNet2DConditionOutput(sample=sample)
            pipe.unet = TracedUNet()

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """
The `get_text_embeds` method takes a text prompt as input and returns the embeddings of that prompt using the `text_encoder`.
        """
        # prompt, negative_prompt: [str]

        # positive
        inputs = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]

        return embeddings

    def predict_noise(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=True, t=None, noise=None):
        if as_latent:
            latents = pred_rgb
        else:
            latents = self.encode_imgs(pred_rgb)

        if t is None:
            t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            if noise is None:
                # add noise
                noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # Save input tensors for UNet
            #torch.save(latent_model_input, "train_latent_model_input.pt")
            #torch.save(t, "train_t.pt")
            #torch.save(text_embeddings, "train_text_embeddings.pt")
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        return noise_pred, t, noise

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, as_latent=False, grad_scale=1):
        """
The `train_step` method computes a single training step for the model. 
It takes text embeddings and an RGB image as inputs, and computes the gradients using the predicted noise residual generated by the unet sub-model. 
It also includes a guidance parameter that can help generate more accurate images.
        """
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # Save input tensors for UNet
            #torch.save(latent_model_input, "train_latent_model_input.pt")
            #torch.save(t, "train_t.pt")
            #torch.save(text_embeddings, "train_text_embeddings.pt")
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        

        # import kiui
        # latents_tmp = torch.randn((1, 4, 64, 64), device=self.device)
        # latents_tmp = latents_tmp.detach()
        # kiui.lo(latents_tmp)
        # self.scheduler.set_timesteps(30)
        # for i, t in enumerate(self.scheduler.timesteps):
        #     latent_model_input = torch.cat([latents_tmp] * 3)
        #     noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']
        #     noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        #     noise_pred = noise_pred_uncond + 10 * (noise_pred_pos - noise_pred_uncond)
        #     latents_tmp = self.scheduler.step(noise_pred, t, latents_tmp)['prev_sample']
        # imgs = self.decode_latents(latents_tmp)
        # kiui.vis.plot_image(imgs)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        grad = grad_scale * w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        return loss

    @torch.no_grad()
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        """
        The `produce_latents` method takes text embeddings and a set of latents as inputs, 
        and produces the corresponding latents for the given text prompts using a generative model.
        """
        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in enumerate(self.scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # Save input tensors for UNet
            #torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
            #torch.save(t, "produce_latents_t.pt")
            #torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):
        """ The `decode_latents` method takes a set of latents as input and generates an RGB image.
        """

        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor

        return latents

    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
        """
        The `prompt_to_img` method takes text prompts and generates a corresponding RGB image. 
        It first gets the text embeddings using the `get_text_embeds` method, 
        then produces the latents using the `produce_latents` method, and finally generates an RGB image using the `decode_latents` method. 
        It also takes in optional parameters such as height and width of the image, the number of inference steps, and negative text prompts.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.imsave('sample.png', imgs[0])
    plt.show()


