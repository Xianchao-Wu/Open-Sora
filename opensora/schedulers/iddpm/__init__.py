from functools import partial

import torch

from opensora.registry import SCHEDULERS

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


@SCHEDULERS.register_module("iddpm")
class IDDPM(SpacedDiffusion):
    def __init__(
        self,
        num_sampling_steps=None, # 100
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000, # 1000
        cfg_scale=4.0, # 7.0
    ):
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type, # <LossType.MSE: 1>
            # rescale_timesteps=rescale_timesteps,
        )

        self.cfg_scale = cfg_scale

    def sample(
        self,
        model, # <class 'opensora.models.stdit.stdit.STDiT'>
        text_encoder, # <class 'opensora.models.text_encoder.t5.T5Encoder'>
        z_size, # (4, 16, 32, 32)
        prompts,
        device,
        additional_args=None,
    ):
        n = len(prompts) # n=1, for 1 prompt
        z = torch.randn(n, *z_size, device=device) # [1, 4, 16, 32, 32]，这是初始噪声的形状
        z = torch.cat([z, z], 0) # torch.Size([2, 4, 16, 32, 32])
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n) # torch.Size([1, 1, 120, 4096]) TODO what is 120?
        model_args["y"] = torch.cat([model_args["y"], y_null], 0) # cat [1, 1, 120, 4096] and [1, 1, 120, 4096] together, the result is now [2, 1, 120, 4096]
        if additional_args is not None:
            model_args.update(additional_args)

        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale) # 1=需要被扩展的函数；2=需要被固定的位置参数；3=需要被固定的关键字参数.
        samples = self.p_sample_loop(
            forward,
            z.shape, # torch.Size([2, 4, 16, 32, 32])
            z,
            clip_denoised=False,
            model_kwargs=model_args, # 一个词典, 'y': [2, 1, 120, 4096], 'mask': [1, 120]
            progress=True,
            device=device, # 'cuda'
        )
        samples, _ = samples.chunk(2, dim=0)
        return samples


def forward_with_cfg(model, x, timestep, y, cfg_scale, **kwargs):
    import ipdb; ipdb.set_trace()
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    half = x[: len(x) // 2] # len(x)=2; shape=torch.Size([1, 4, 16, 32, 32])
    combined = torch.cat([half, half], dim=0) # torch.Size([2, 4, 16, 32, 32])
    model_out = model.forward(combined, timestep, y, **kwargs) # NOTE important!
    import ipdb; ipdb.set_trace()
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)

