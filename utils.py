import os
import sys
sys.path.insert(0, 'jaxtorch')
import math
import io
import requests
import jax
import jax.numpy as jnp
from jaxtorch import nn, init


## Define the model (a residual U-Net)
class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return self.main(cx, input) + self.skip(cx, input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(p=0.1) if dropout else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(p=0.1) if dropout else nn.Identity(),
            nn.ReLU(),
        ], skip)


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, cx, input):
        return jnp.concatenate([self.main(cx, input), self.skip(cx, input)], axis=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = init.normal(out_features // 2, in_features, stddev=std)

    def forward(self, cx, input):
        f = 2 * math.pi * input @ cx[self.weight].T
        return jnp.concatenate([f.cos(), f.sin()], axis=-1)


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.class_embed = nn.Embedding(10, 4)

        self.net = nn.Sequential(
            ResConvBlock(3 + 16 + 4, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.image.Downsample2d(),  # 64x64 -> 32x32
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.image.Downsample2d(),  # 32x32 -> 16x16
                    ResConvBlock(c * 2, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    SkipBlock([
                        nn.image.Downsample2d(),  # 16x16 -> 8x8
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        ResConvBlock(c * 2, c * 2, c * 2),
                        nn.image.Upsample2d(),
                    ]),
                    ResConvBlock(c * 4, c * 2, c * 2),
                    ResConvBlock(c * 2, c * 2, c * 2),
                    nn.image.Upsample2d(),
                ]),
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.image.Upsample2d(),  # Haven't implemented ConvTranpose2d yet.
            ]),
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout=False),
        )

    def forward(self, cx, input, log_snrs, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(cx, log_snrs[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cx, cond), input.shape)
        return self.net(cx, jnp.concatenate([input, class_embed, timestep_embed], axis=1))


def expand_to_planes(input, shape):
    return input[..., None, None].broadcast_to(input.shape[:2] + shape[2:])


# Define the noise schedule

def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -jnp.expm1(1e-4 + 10 * t ** 2).log()


def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image and for the noise, given
    the log SNR for a timestep."""
    alphas_squared = jax.nn.sigmoid(log_snrs)
    sigmas_squared = jax.nn.sigmoid(-log_snrs)
    return alphas_squared.sqrt(), sigmas_squared.sqrt()


## Define additional functions

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def fetch_model(url_or_path):
    basename = os.path.basename(url_or_path)
    if os.path.exists(basename):
        return basename
    else:

        data = fetch(url_or_path).read()
        with open(basename, 'wb') as fp:
            fp.write(data)
        return basename


def Normalize(mean, std):
    mean = jnp.array(mean).reshape(3, 1, 1)
    std = jnp.array(std).reshape(3, 1, 1)

    def forward(image):
        return (image - mean) / std

    return forward


def norm1(x):
    """Normalize to the unit sphere."""
    return x / x.square().sum(axis=-1, keepdims=True).sqrt()


def spherical_dist_loss(x, y):
    x = norm1(x)
    y = norm1(y)
    return (x - y).square().sum(axis=-1).sqrt().div(2).arcsin().square().mul(2)
