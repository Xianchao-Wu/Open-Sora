import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from einops import rearrange

from opensora.registry import MODELS


@MODELS.register_module()
class VideoAutoencoderKL(nn.Module):
    def __init__(self, from_pretrained=None, micro_batch_size=None):
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(from_pretrained)
        self.out_channels = self.module.config.latent_channels # 4
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size # 4

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        else:
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.encode(x_bs).latent_dist.sample().mul_(0.18215)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x):
        # x: (B, C, T, H, W), e.g., torch.Size([1, 4, 16, 32, 32])
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W") # torch.Size([16, 4, 32, 32])
        if self.micro_batch_size is None:
            x = self.module.decode(x / 0.18215).sample
        else:
            bs = self.micro_batch_size # 4 NOTE
            x_out = []
            for i in range(0, x.shape[0], bs): # 0 to 16, with step=4
                x_bs = x[i : i + bs] # torch.Size([4, 4, 32, 32])
                x_bs = self.module.decode(x_bs / 0.18215).sample # NOTE
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0) # torch.Size([16, 3, 256, 256]) = x.shape
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x # torch.Size([1, 3, 16, 256, 256]) NOTE

    def get_latent_size(self, input_size):
        for i in range(3):
            assert input_size[i] % self.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // self.patch_size[i] for i in range(3)]
        return input_size


@MODELS.register_module()
class VideoAutoencoderKLTemporalDecoder(nn.Module):
    def __init__(self, from_pretrained=None):
        super().__init__()
        self.module = AutoencoderKLTemporalDecoder.from_pretrained(from_pretrained)
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)

    def encode(self, x):
        raise NotImplementedError

    def decode(self, x):
        B, _, T = x.shape[:3]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        x = self.module.decode(x / 0.18215, num_frames=T).sample
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        for i in range(3):
            assert input_size[i] % self.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // self.patch_size[i] for i in range(3)]
        return input_size
