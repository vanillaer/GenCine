"""Bidirectional denoising and sampling."""
import torch
from einops import rearrange
from tqdm import tqdm

from .models.wan_dit import sinusoidal_embedding_1d
from .scheduler import FlowMatchScheduler


def model_fn_wan_video(dit, vace, scale3d, latents, timestep, context,
                       clip_feature, y, reference_latents, vace_context,
                       scale_cond, vace_scale=1.0):
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    t_mod = t_mod + scale3d(scale_cond).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    x = torch.cat([latents, y], dim=1)
    clip_embedding = dit.img_emb(clip_feature)
    context = torch.cat([clip_embedding, context], dim=1)
    x = dit.patchify(x)
    f, h, w = x.shape[2:]
    f_video = f
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    reference_latents = dit.ref_conv(reference_latents[:, :, 0]).flatten(2).transpose(1, 2)
    reference_prefix_frames = reference_latents.shape[1] // (h * w)
    assert reference_prefix_frames == 1
    x = torch.concat([reference_latents, x], dim=1)
    f += reference_prefix_frames
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    hints = vace(x, vace_context, context, t_mod, freqs,
                 main_grid_size_full=(f, h, w),
                 main_grid_size_video=(f_video, h, w),
                 main_prefix_frames=reference_prefix_frames)
    for block_id, block in enumerate(dit.blocks):
        x = block(x, context, t_mod, freqs)
        if block_id in vace.vace_layers_mapping:
            x = x + hints[vace.vace_layers_mapping[block_id]] * vace_scale
    x = dit.head(x, t)
    x = x[:, reference_latents.shape[1]:]
    return dit.unpatchify(x, (f_video, h, w))


def generate_noise(shape, seed, device):
    generator = torch.Generator(device).manual_seed(seed)
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(torch.bfloat16)


@torch.no_grad()
def generate(models, inputs, settings, device='cuda', seed=None, save_validation=False):
    """Return RGB and optional comparison tensors; reuse models across examples."""
    dit, vace, scale3d, vae = (models[k] for k in ('dit', 'vace', 'scale3d', 'vae_decoder'))
    for model in (dit, vace, scale3d):
        model.to(device)
    vae.cpu()
    scheduler = FlowMatchScheduler()
    scheduler.set_timesteps(settings['num_inference_steps'], settings['denoising_strength'], settings['sigma_shift'])
    inputs = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in inputs.items()}
    latents = inputs.pop('initial_noise')
    if seed is not None:
        latents = generate_noise(latents.shape, seed, device)
    context = inputs.pop('context')
    negative_context = inputs.pop('negative_context', None)
    trace = {}
    if save_validation:
        trace = {'initial_noise': latents.cpu(), 'sigmas': scheduler.sigmas, 'timesteps': scheduler.timesteps}
    for i, timestep in enumerate(tqdm(scheduler.timesteps, desc='denoise')):
        t = timestep.unsqueeze(0).to(dtype=torch.bfloat16, device=device)
        positive = model_fn_wan_video(dit, vace, scale3d, latents, t, context,
                                      **inputs, vace_scale=settings['vace_scale'])
        if settings['cfg_scale'] != 1.0:
            assert negative_context is not None, 'CFG requires an encoded negative prompt'
            negative = model_fn_wan_video(dit, vace, scale3d, latents, t, negative_context,
                                          **inputs, vace_scale=settings['vace_scale'])
            prediction = negative + settings['cfg_scale'] * (positive - negative)
            if save_validation and i == 0:
                trace['first_negative_prediction'] = negative.cpu()
        else:
            prediction = positive
        latents = scheduler.step(prediction, timestep, latents)
        if save_validation and i == 0:
            trace.update(first_positive_prediction=positive.cpu(), first_prediction=prediction.cpu(), first_latent=latents.cpu())
    assert torch.isfinite(latents).all(), 'Nonfinite final latent'
    if save_validation:
        trace['final_latent'] = latents.cpu()
    # A single explicit phase transfer keeps untiled decoding within 80 GB.
    for model in (dit, vace, scale3d):
        model.cpu()
    torch.cuda.empty_cache()
    vae.to(device)
    rgb = vae.decode(latents, device=device)
    assert torch.isfinite(rgb).all(), 'Nonfinite decoded RGB'
    vae.cpu()
    torch.cuda.empty_cache()
    return rgb, trace
