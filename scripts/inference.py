import os

import torch
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from colossalai.cluster import DistCoordinator


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False)
    print(cfg)

    # init distributed
    colossalai.launch_from_torch({})
    import ipdb; ipdb.set_trace() # NOTE
    coordinator = DistCoordinator()

    if coordinator.world_size > 1:
        set_sequence_parallel_group(dist.group.WORLD) 
        enable_sequence_parallelism = True
    else:
        enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size) # [16, 32, 32]
    import ipdb; ipdb.set_trace()
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
    import ipdb; ipdb.set_trace()
    model = build_module(
        cfg.model, # NOTE very important
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=enable_sequence_parallelism,
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance
    
    import ipdb; ipdb.set_trace()
    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    import ipdb; ipdb.set_trace()
    # 3.3. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)

    import ipdb; ipdb.set_trace()
    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    save_dir = cfg.save_dir # './outputs/samples/'
    os.makedirs(save_dir, exist_ok=True)
    for i in range(0, len(prompts), cfg.batch_size): # cfg.batch_size=1
        import ipdb; ipdb.set_trace()
        batch_prompts = prompts[i : i + cfg.batch_size]
        samples = scheduler.sample(
            model, # <class 'opensora.models.stdit.stdit.STDiT'>
            text_encoder, # <opensora.models.text_encoder.t5.T5Encoder object at 0x7f8618367d90>
            z_size=(vae.out_channels, *latent_size), # vae.out_channels=4, latent_size=[16, 32, 32]
            prompts=batch_prompts,
            device=device, # 'cuda'
            additional_args=model_args, # {}
        )
        import ipdb; ipdb.set_trace()
        samples = vae.decode(samples.to(dtype))

        if coordinator.is_master():
            for idx, sample in enumerate(samples):
                print(f"Prompt: {batch_prompts[idx]}")
                save_path = os.path.join(save_dir, f"sample_{sample_idx}")
                save_sample(sample, fps=cfg.fps, save_path=save_path)
                sample_idx += 1


if __name__ == "__main__":
    main()
