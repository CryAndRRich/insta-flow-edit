from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.sd3 import StableDiffusion3Base, register_sampler


@register_sampler(name="flowalign")
class SD3FlowAlign(StableDiffusion3Base):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 50,
               n_start: int = 0,
               cfg_scale: float = 7.0,
               zeta: float = 0.01,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None,
               neg_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)

        with torch.no_grad():
            z_src = self.encode(src_img)

        x_t = z_src.clone()

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(timesteps, total=NFE, desc="SD3 FlowAlign")
        for i, t in enumerate(pbar):
            if i < n_start:
                continue

            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = sigma_next - sigma

            eps = torch.randn_like(z_src)
            qt = (1 - sigma) * z_src + sigma * eps
            pt = x_t + qt - z_src

            with torch.no_grad():
                v_tar_cond = self.predict_vector(pt, t, tgt_emb, tgt_pool)
                v_tar_src = self.predict_vector(pt, t, src_emb, src_pool)
                v_tar = v_tar_src + cfg_scale * (v_tar_cond - v_tar_src)

                v_src = self.predict_vector(qt, t, src_emb, src_pool)

            reg_term = (qt - pt) + sigma * (v_tar - v_src)
            x_t = x_t + dt * (v_tar - v_src) + zeta * reg_term

        with torch.no_grad():
            img = self.decode(x_t)
        return img