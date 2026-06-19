from __future__ import annotations

import unittest

import torch

from scripts.model import (
    DDPM,
    ConditionalUNet,
    DiffusionConfig,
    ModelConfig,
    images_to_uint8,
    model_config_from_dict,
    model_config_to_dict,
)


class ModelModuleTests(unittest.TestCase):
    def test_forward_shape_matches_input(self) -> None:
        model = ConditionalUNet(self._model_config())
        x = torch.randn(2, 3, 64, 64)
        t = torch.tensor([0, 1])
        animals = torch.tensor([0, 1])
        objects = torch.tensor([2, 3])
        self.assertEqual(tuple(model(x, t, animals, objects).shape), tuple(x.shape))

    def test_training_loss_is_finite_and_backpropagates(self) -> None:
        model = ConditionalUNet(self._model_config())
        diffusion = DDPM(DiffusionConfig(timesteps=4))
        images = torch.randn(2, 3, 64, 64)
        animals = torch.tensor([0, 1])
        objects = torch.tensor([2, 3])

        loss = diffusion.training_loss(model, images, animals, objects)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(any(torch.isfinite(g).all().item() and g.abs().sum().item() > 0 for g in grads))

    def test_schedule_is_finite_and_monotonic(self) -> None:
        diffusion = DDPM(DiffusionConfig(timesteps=8))
        self.assertTrue(torch.isfinite(diffusion.betas).all().item())
        self.assertTrue(torch.isfinite(diffusion.alpha_bars).all().item())
        self.assertTrue(torch.all(diffusion.betas[1:] >= diffusion.betas[:-1]).item())
        self.assertTrue(torch.all(diffusion.alpha_bars[1:] <= diffusion.alpha_bars[:-1]).item())

    def test_sampler_returns_rgb_tensor_shape(self) -> None:
        torch.manual_seed(0)
        model = ConditionalUNet(self._model_config())
        diffusion = DDPM(DiffusionConfig(timesteps=2))
        samples = diffusion.sample(
            model,
            torch.tensor([0]),
            torch.tensor([0]),
            image_size=64,
            sampling_steps=2,
            guidance_scale=1.5,
        )
        self.assertEqual(tuple(samples.shape), (1, 3, 64, 64))
        self.assertEqual(images_to_uint8(samples).dtype, torch.uint8)

    def test_model_config_round_trips_tuple_fields(self) -> None:
        config = self._model_config()
        loaded = model_config_from_dict(model_config_to_dict(config))
        self.assertEqual(loaded.channel_mults, config.channel_mults)
        self.assertEqual(loaded.attention_resolutions, config.attention_resolutions)

    def _model_config(self) -> ModelConfig:
        return ModelConfig(
            base_channels=4,
            channel_mults=(1, 2, 2),
            num_res_blocks=1,
            attention_resolutions=(16,),
            attention_heads=2,
            cond_dim=32,
            dropout=0.0,
            condition_dropout=0.0,
        )


if __name__ == "__main__":
    unittest.main()
