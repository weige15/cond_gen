from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from scripts.data import ANIMAL_LABELS, IMAGE_SIZE, OBJECT_LABELS


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = IMAGE_SIZE
    in_channels: int = 3
    base_channels: int = 128
    channel_mults: tuple[int, ...] = (1, 2, 3, 4)
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (16, 8)
    attention_heads: int = 4
    cond_dim: int = 512
    dropout: float = 0.1
    condition_dropout: float = 0.1
    num_animals: int = len(ANIMAL_LABELS)
    num_objects: int = len(OBJECT_LABELS)


@dataclass(frozen=True)
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"


def model_config_to_dict(config: ModelConfig) -> dict[str, Any]:
    return asdict(config)


def diffusion_config_to_dict(config: DiffusionConfig) -> dict[str, Any]:
    return asdict(config)


def model_config_from_dict(values: dict[str, object]) -> ModelConfig:
    values = dict(values)
    for key in ("channel_mults", "attention_resolutions"):
        if key in values and not isinstance(values[key], tuple):
            values[key] = tuple(values[key])
    return ModelConfig(**values)


def diffusion_config_from_dict(values: dict[str, object]) -> DiffusionConfig:
    return DiffusionConfig(**values)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConditionalUNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if not config.channel_mults:
            raise ValueError("channel_mults must not be empty")
        if config.num_res_blocks < 1:
            raise ValueError("num_res_blocks must be >= 1")
        divisor = 2 ** (len(config.channel_mults) - 1)
        if config.image_size % divisor:
            raise ValueError(f"image_size must be divisible by {divisor}")

        self.config = config
        cond_dim = config.cond_dim
        channels = [config.base_channels * mult for mult in config.channel_mults]
        resolutions = [config.image_size // (2**idx) for idx in range(len(channels))]

        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )
        self.animal_emb = nn.Embedding(config.num_animals, cond_dim)
        self.object_emb = nn.Embedding(config.num_objects, cond_dim)
        self.pair_emb = nn.Embedding(config.num_animals * config.num_objects, cond_dim)
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))
        self.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, cond_dim))

        self.in_conv = nn.Conv2d(config.in_channels, channels[0], 3, padding=1)
        self.down_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = channels[0]
        for level, (out_ch, resolution) in enumerate(zip(channels, resolutions, strict=True)):
            layers: list[nn.Module] = []
            for block_idx in range(config.num_res_blocks):
                layers.append(ResBlock(in_ch if block_idx == 0 else out_ch, out_ch, cond_dim, config.dropout))
                if resolution in config.attention_resolutions:
                    layers.append(AttentionBlock(out_ch, config.attention_heads))
            self.down_levels.append(CondSequential(*layers))
            in_ch = out_ch
            if level < len(channels) - 1:
                self.downsamples.append(nn.Conv2d(out_ch, channels[level + 1], 3, stride=2, padding=1))
                in_ch = channels[level + 1]

        mid_ch = channels[-1]
        self.mid = CondSequential(
            ResBlock(mid_ch, mid_ch, cond_dim, config.dropout),
            AttentionBlock(mid_ch, config.attention_heads),
            ResBlock(mid_ch, mid_ch, cond_dim, config.dropout),
        )

        self.up_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        current_ch = mid_ch
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            resolution = resolutions[level]
            layers = [ResBlock(current_ch + out_ch, out_ch, cond_dim, config.dropout)]
            if resolution in config.attention_resolutions:
                layers.append(AttentionBlock(out_ch, config.attention_heads))
            for _ in range(config.num_res_blocks - 1):
                layers.append(ResBlock(out_ch, out_ch, cond_dim, config.dropout))
                if resolution in config.attention_resolutions:
                    layers.append(AttentionBlock(out_ch, config.attention_heads))
            self.up_levels.append(CondSequential(*layers))
            current_ch = out_ch
            if level > 0:
                self.upsamples.append(nn.ConvTranspose2d(out_ch, channels[level - 1], 4, stride=2, padding=1))
                current_ch = channels[level - 1]

        self.out = nn.Sequential(_group_norm(channels[0]), nn.SiLU(), nn.Conv2d(channels[0], config.in_channels, 3, padding=1))

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        animal_ids: torch.Tensor,
        object_ids: torch.Tensor,
        *,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected image tensor with 4 dims, got {x.shape}")

        cond = self._conditioning(timesteps, animal_ids, object_ids, force_uncond=force_uncond)
        h = self.in_conv(x)
        skips = []
        for level, block in enumerate(self.down_levels):
            h = block(h, cond)
            skips.append(h)
            if level < len(self.downsamples):
                h = self.downsamples[level](h)

        h = self.mid(h, cond)
        for idx, block in enumerate(self.up_levels):
            skip = skips.pop()
            h = block(torch.cat([h, skip], dim=1), cond)
            if idx < len(self.upsamples):
                h = self.upsamples[idx](h)
        return self.out(h)

    def _conditioning(
        self,
        timesteps: torch.Tensor,
        animal_ids: torch.Tensor,
        object_ids: torch.Tensor,
        *,
        force_uncond: bool,
    ) -> torch.Tensor:
        time = self.time_mlp(timestep_embedding(timesteps, self.config.cond_dim))
        pair_ids = animal_ids * self.config.num_objects + object_ids
        label_cond = self.animal_emb(animal_ids) + self.object_emb(object_ids) + self.pair_emb(pair_ids)

        if force_uncond:
            label_cond = self.null_cond[None].expand_as(label_cond)
        elif self.training and self.config.condition_dropout > 0:
            drop = torch.rand(label_cond.shape[0], device=label_cond.device) < self.config.condition_dropout
            label_cond = torch.where(drop[:, None], self.null_cond[None], label_cond)

        return self.cond_mlp(time + label_cond)


class CondSequential(nn.Sequential):
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if getattr(layer, "uses_condition", False):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class ResBlock(nn.Module):
    uses_condition = True

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, dropout: float):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.cond = nn.Linear(cond_dim, out_channels * 2)
        self.norm2 = _group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.cond(cond)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.heads = _attention_heads(channels, heads)
        self.norm = _group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        head_dim = channels // self.heads
        q = q.reshape(batch, self.heads, head_dim, height * width)
        k = k.reshape(batch, self.heads, head_dim, height * width)
        v = v.reshape(batch, self.heads, head_dim, height * width)
        scale = head_dim**-0.5
        attn = torch.softmax(torch.einsum("bhcn,bhcm->bhnm", q * scale, k), dim=-1)
        h = torch.einsum("bhnm,bhcm->bhcn", attn, v).reshape(batch, channels, height * width)
        return x + self.proj(h).reshape(batch, channels, height, width)


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _attention_heads(channels: int, requested_heads: int) -> int:
    for heads in range(max(requested_heads, 1), 0, -1):
        if channels % heads == 0:
            return heads
    return 1


class DDPM:
    def __init__(self, config: DiffusionConfig):
        if config.timesteps < 1:
            raise ValueError("timesteps must be >= 1")
        self.config = config
        betas = self._make_betas(config)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = alpha_bars_prev
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        self.posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

    def q_sample(self, clean_images: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._extract(self.sqrt_alpha_bars, timesteps, clean_images.shape) * clean_images + self._extract(
            self.sqrt_one_minus_alpha_bars, timesteps, clean_images.shape
        ) * noise

    def training_loss(
        self,
        model: nn.Module,
        clean_images: torch.Tensor,
        animal_ids: torch.Tensor,
        object_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch = clean_images.shape[0]
        timesteps = torch.randint(0, self.config.timesteps, (batch,), device=clean_images.device)
        noise = torch.randn_like(clean_images)
        noisy = self.q_sample(clean_images, timesteps, noise)
        predicted = model(noisy, timesteps, animal_ids, object_ids)
        return F.mse_loss(predicted, noise)

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        animal_ids: torch.Tensor,
        object_ids: torch.Tensor,
        *,
        image_size: int = IMAGE_SIZE,
        sampling_steps: int | None = None,
        guidance_scale: float = 1.0,
        eta: float = 0.0,
    ) -> torch.Tensor:
        if sampling_steps is None:
            sampling_steps = self.config.timesteps
        if not 1 <= sampling_steps <= self.config.timesteps:
            raise ValueError(f"sampling_steps must be in [1, {self.config.timesteps}]")

        model_device = next(model.parameters()).device
        animal_ids = animal_ids.to(model_device)
        object_ids = object_ids.to(model_device)
        x = torch.randn((animal_ids.shape[0], 3, image_size, image_size), device=model_device)
        steps = torch.linspace(self.config.timesteps - 1, 0, sampling_steps, device=model_device).long()

        for idx, step in enumerate(steps):
            timesteps = torch.full((animal_ids.shape[0],), int(step.item()), device=model_device, dtype=torch.long)
            alpha_bar_t = self._extract(self.alpha_bars, timesteps, x.shape)
            eps = self._predict_noise(model, x, timesteps, animal_ids, object_ids, guidance_scale)
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

            if idx == len(steps) - 1:
                x = pred_x0
                continue

            prev_step = int(steps[idx + 1].item())
            prev_timesteps = torch.full_like(timesteps, prev_step)
            alpha_bar_prev = self._extract(self.alpha_bars, prev_timesteps, x.shape)
            sigma = eta * torch.sqrt(
                torch.clamp((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev), min=0.0)
            )
            direction = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.0)) * eps
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction + sigma * noise
        return x.clamp(-1.0, 1.0)

    def _predict_noise(
        self,
        model: nn.Module,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        animal_ids: torch.Tensor,
        object_ids: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        if guidance_scale == 1.0:
            return model(x, timesteps, animal_ids, object_ids)
        eps_uncond = model(x, timesteps, animal_ids, object_ids, force_uncond=True)
        eps_cond = model(x, timesteps, animal_ids, object_ids)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.to(timesteps.device).gather(0, timesteps)
        return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))

    def _make_betas(self, config: DiffusionConfig) -> torch.Tensor:
        if config.schedule == "linear":
            return torch.linspace(config.beta_start, config.beta_end, config.timesteps, dtype=torch.float32)
        if config.schedule != "cosine":
            raise ValueError(f"unknown diffusion schedule: {config.schedule}")

        steps = torch.arange(config.timesteps + 1, dtype=torch.float64)
        s = 0.008
        alpha_bars = torch.cos(((steps / config.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        return betas.clamp(1e-5, 0.999).float()


def build_model(config: ModelConfig) -> ConditionalUNet:
    return ConditionalUNet(config)


def build_diffusion(config: DiffusionConfig) -> DDPM:
    return DDPM(config)


def images_to_uint8(images: torch.Tensor) -> torch.Tensor:
    return ((images.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
