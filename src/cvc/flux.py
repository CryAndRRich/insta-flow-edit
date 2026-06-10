from typing import Tuple, Optional
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from ..base.flux import FluxBase, register_sampler
from ..base.helper import calculate_shift


@register_sampler(name="cvc")
class FluxCVC(FluxBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               NFE: int = 28,
               alpha: float = 1.0,
               beta: float = 3.5,
               eta: float = 0.2,
               cfg_scale: float = 3.5,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool, src_txt_ids = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool, tgt_txt_ids = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            z_src, img_ids, h_lat, w_lat = self.prepare_latents(src_img)

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
        guidance = torch.tensor([cfg_scale], device=device).expand(z_src.shape[0])

        z_edit = z_src.clone()

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="FLUX CVC")
        for i, t in pbar:
            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0).to(device)
            dt = t_next - t_curr
            t_tensor = t_curr.view(1)

            eps = torch.randn_like(z_src)
            qt = (1 - t_curr) * z_src + t_curr * eps
            pt = z_edit + qt - z_src

            with torch.no_grad():
                v1 = self.predict_vector(qt, t_tensor, src_emb, src_pool, src_txt_ids, img_ids, guidance=guidance)
                v2 = self.predict_vector(pt, t_tensor, src_emb, src_pool, src_txt_ids, img_ids, guidance=guidance)
                v3 = self.predict_vector(pt, t_tensor, tgt_emb, tgt_pool, tgt_txt_ids, img_ids, guidance=guidance)

            v_delta = alpha * (v2 - v1) + beta * (v3 - v2)

            # Tweedie posterior gradient correction (Algorithm 1 lines 13-15)
            # L_align = ||v_delta * dt - z_src||^2  =>  grad = 2 * dt * (v_delta*dt - z_src)
            dx = v_delta * dt
            grad = 2.0 * dt * (dx - z_src)
            v_new = v_delta - eta * grad

            z_edit = z_edit + eta * dt * v_new

        with torch.no_grad():
            img = self.decode(z_edit, h_lat, w_lat)
        return img