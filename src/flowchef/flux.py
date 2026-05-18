from typing import Tuple, Optional
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from ..base.flux import FluxBase, register_sampler
from ..base.helper import calculate_shift


@register_sampler(name="flowchef")
class FluxFlowChef(FluxBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               NFE: int = 28,
               guidance_scale: float = 7.5,
               cfg_scale: float = 5.5,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        """
        FlowChef for FLUX (arXiv:2412.00100): steering rectified flow via
        loss-based gradient guidance with gradient skipping.
        Starts from pure noise — no inversion. CFG via FLUX guidance distillation.
        """
        with torch.no_grad():
            tgt_emb, tgt_pool, tgt_txt_ids = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            z_src, img_ids, h_lat, w_lat = self.prepare_latents(src_img)

        x_t = torch.randn_like(z_src)

        sigmas = np.linspace(1.0, 1 / NFE, NFE)
        mu = calculate_shift(
            z_src.shape[1],
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler, NFE, self.device, timesteps=None, sigmas=sigmas, mu=mu
        )

        device = self.device if not self.offload else "cuda"
        guidance = torch.tensor([cfg_scale], device=device).expand(x_t.shape[0])

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="FLUX FlowChef")
        for i, t in pbar:
            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0).to(device)
            dt = t_next - t_curr
            t_tensor = t_curr.view(1)

            with torch.no_grad():
                v = self.predict_vector(x_t, t_tensor, tgt_emb, tgt_pool, tgt_txt_ids, img_ids, guidance=guidance)

            x0_hat = x_t - t_curr * v
            loss_grad = x0_hat - z_src
            x_t = x_t + dt * v - guidance_scale * loss_grad

        with torch.no_grad():
            img = self.decode(x_t, h_lat, w_lat)
        return img
