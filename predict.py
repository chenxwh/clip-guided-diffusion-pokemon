import os
import sys

sys.path.insert(0, 'jaxtorch')
sys.path.insert(0, 'CLIP_JAX')
import tempfile
from pathlib import Path
import numpy as np
import torch
from torchvision import utils
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import torch.utils.data
from functools import partial
from PIL import Image
import cog
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jaxtorch
from jaxtorch import PRNG, Context
from jaxtorch import nn, init
import clip_jax
from utils import *


class StateDict(dict):
    pass


sys.modules["__main__"].StateDict = StateDict

model = Diffusion()
params_ema = model.init_weights(jax.random.PRNGKey(0))
state_dict = jaxtorch.pt.load(
    fetch_model('https://set.zlkj.in/models/diffusion/pokemon_diffusion_gen3+4_c64_6783.pth'))
model.load_state_dict(params_ema, state_dict['model_ema'], strict=False)

image_fn, text_fn, clip_params, _ = clip_jax.load('ViT-B/32')


class Predictor(cog.Predictor):

    def setup(self):
        print(f'Using device: {jax.devices()}')
        print(f'Model parameters: {sum(np.prod(p.shape) for p in params_ema.values.values())}')

    @cog.input(
        "prompt",
        type=str,
        default='a pokemon resembling ♲ #pixelart',
        help="prompt for generating image"
    )
    def predict(self, prompt='a pokemon resembling ♲ #pixelart'):

        def base_cond_fn(x, t, text_embed, clip_guidance_scale, classes, key, params_ema, clip_params):
            rng = PRNG(key)
            n = x.shape[0]

            log_snrs = get_ddpm_schedule(t)
            alphas, sigmas = get_alphas_sigmas(log_snrs)
            normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                  std=[0.26862954, 0.26130258, 0.27577711])

            def denoise(x, key):
                eps = eval_model(params_ema, x, log_snrs.broadcast_to([n]), classes, rng.split())
                # Predict the denoised image
                pred = (x - eps * sigmas) / alphas
                x_in = pred * sigmas + x * alphas
                return x_in

            x_in, backward = jax.vjp(partial(denoise, key=rng.split()), x)

            def clip_loss(x_in):
                x_in = jax.image.resize(x_in, [n, 3, 224, 224], method='nearest')
                clip_in = normalize(x_in.add(1).div(2))
                image_embeds = emb_image(clip_in, clip_params).reshape([n, 512])
                losses = spherical_dist_loss(image_embeds, text_embed)
                return losses.sum() * clip_guidance_scale

            clip_grad = jax.grad(clip_loss)(x_in)

            return -backward(clip_grad)[0]

        base_cond_fn = jax.jit(base_cond_fn)

        ## Settings for the run
        seed = 0

        # Prompt for CLIP guidance
        text_embed = txt(prompt)

        # Strength of conditioning
        clip_guidance_scale = 2000

        eta = 1.0
        batch_size = 16

        # Image size. Was trained on 64x64. Must be a multiple of 8 but different sizes are possible.
        image_size = 64

        # Number of steps for sampling, more = better quality generally
        steps = 250

        # demo()
        tqdm.write('Sampling...')
        rng = PRNG(jax.random.PRNGKey(seed))

        fakes = jax.random.normal(rng.split(), [batch_size, 3, image_size, image_size])

        fakes_classes = jnp.array([0] * batch_size)  # plain
        ts = jnp.ones([batch_size])

        # Create the noise schedule
        t = jnp.linspace(1, 0, steps + 1)[:-1]
        log_snrs = get_ddpm_schedule(t)
        alphas, sigmas = get_alphas_sigmas(log_snrs)

        out_path = Path(tempfile.mkdtemp()) / "out.png"
        os.makedirs('res', exist_ok=True)

        # The sampling loop
        for i in range(steps):

            eps = eval_model(params_ema, fakes, ts * log_snrs[i], fakes_classes, rng.split())
            # Predict the denoised image
            pred = (fakes - eps * sigmas[i]) / alphas[i]

            # If we are not on the last timestep, compute the noisy image for the
            # next timestep.
            if i < steps - 1:
                # cond_fn() is just calling base_cond_fn()
                cond_score = base_cond_fn(fakes, t[i], text_embed, clip_guidance_scale,
                                          fakes_classes, rng.split(), params_ema, clip_params)

                eps = eps - sigmas[i] * cond_score
                pred = (fakes - eps * sigmas[i]) / alphas[i]

                # If eta > 0, adjust the scaling factor for the predicted noise
                # downward according to the amount of additional noise to add
                ddim_sigma = eta * (sigmas[i + 1] ** 2 / sigmas[i] ** 2).sqrt() * (
                        1 - alphas[i] ** 2 / alphas[i + 1] ** 2).sqrt()
                adjusted_sigma = (sigmas[i + 1] ** 2 - ddim_sigma ** 2).sqrt()

                # Recombine the predicted noise and predicted denoised image in the
                # correct proportions for the next step
                fakes = pred * alphas[i + 1] + eps * adjusted_sigma

                # Add the correct amount of fresh noise
                if eta:
                    fakes += jax.random.normal(rng.split(), fakes.shape) * ddim_sigma

            # If we are on the last timestep, output the denoised image
            else:
                fakes = pred

            if i > 0 and i % 10 == 0:
                # yield checkin(i, fakes, steps, out_path)
                yield checkin(i, fakes, steps, out_path)

        grid = utils.make_grid(torch.tensor(np.array(fakes)), 4).cpu()
        upscale = T.Resize(522, interpolation=Image.NEAREST)
        TF.to_pil_image(upscale(grid.add(1).div(2).clamp(0, 1))).save(str(out_path))

        return out_path


def checkin(i, fakes, steps, out_path):
    tqdm.write(f'step: {i}, out of {steps} steps')
    grid = utils.make_grid(torch.tensor(np.array(fakes)), 4).cpu()
    upscale = T.Resize(522, interpolation=Image.NEAREST)
    TF.to_pil_image(upscale(grid.add(1).div(2).clamp(0, 1))).save(str(out_path))
    return out_path

## Define model wrappers
@jax.jit
def eval_model(params, xs, ts, classes, key):
    cx = Context(params, key).eval_mode_()
    return model(cx, xs, ts, classes)


def txt(prompt):
    """Returns normalized embedding."""
    text = clip_jax.tokenize([prompt])
    text_embed = text_fn(clip_params, text)
    return norm1(text_embed.reshape(512))


def emb_image(image, clip_params=None):
    return norm1(image_fn(clip_params, image))
