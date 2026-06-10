from typing import Tuple, Optional
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from ..base.flux import FluxBase, register_sampler
from ..base.helper import calculate_shift


@register_sampler(name="flowalign")
class FluxFlowAlign(FluxBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               NFE: int = 28,
               n_start: int = 0,
               cfg_scale: float = 3.5,
               zeta: float = 0.01,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool, src_txt_ids = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool, tgt_txt_ids = self.prepare_embed(tgt_prompt, tgt_prompt_emb)

        with torch.no_grad():
            z_src, img_ids, h_lat, w_lat = self.prepare_latents(src_img)

        x_t = z_src.clone()

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
        # FLUX uses guidance distillation: pass cfg_scale directly to the transformer
        # rather than computing an external CFG combination, matching the FlowEdit
        # convention for FLUX in this codebase.
        tar_guidance = torch.tensor([cfg_scale], device=device).expand(z_src.shape[0])
        src_guidance = torch.tensor([1.0], device=device).expand(z_src.shape[0])

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="FLUX FlowAlign")
        for i, t in pbar:
            if i < n_start:
                continue

            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0.0).to(device)
            dt = t_next - t_curr
            t_tensor = t_curr.view(1)

            eps = torch.randn_like(z_src)
            qt = (1 - t_curr) * z_src + t_curr * eps
            pt = x_t + qt - z_src

            with torch.no_grad():
                # For FLUX, guidance distillation is handled by the transformer
                # internally via the guidance parameter. v_tar uses tar_guidance
                # (cfg_scale) and v_src uses src_guidance (1.0), consistent with
                # how FlowEdit is adapted for FLUX in this codebase.
                v_tar = self.predict_vector(pt, t_tensor, tgt_emb, tgt_pool, tgt_txt_ids, img_ids, guidance=tar_guidance)
                v_src = self.predict_vector(qt, t_tensor, src_emb, src_pool, src_txt_ids, img_ids, guidance=src_guidance)

            reg_term = (qt - pt) + t_curr * (v_tar - v_src)
            x_t = x_t + dt * (v_tar - v_src) + zeta * reg_term

        with torch.no_grad():
            img = self.decode(x_t, h_lat, w_lat)
        return img