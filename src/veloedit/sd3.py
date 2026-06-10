from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.sd3 import StableDiffusion3Base, register_sampler


@register_sampler(name="veloedit")
class SD3VeloEdit(StableDiffusion3Base):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 50,
               N: int = 1,
               tau: float = 0.4,
               alpha: float = 0.8,
               cfg_scale: float = 13.5,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None,
               neg_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            neg_emb, neg_pool = self.prepare_embed(neg_prompt, neg_prompt_emb)
            z_src = self.encode(src_img)

        x_t = torch.randn_like(z_src)

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="SD3 VeloEdit")
        for i, t in pbar:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = sigma_next - sigma

            with torch.no_grad():
                v_cond = self.predict_vector(x_t, t, tgt_emb, tgt_pool)
                v_neg  = self.predict_vector(x_t, t, neg_emb, neg_pool)
                v_pred = v_neg + cfg_scale * (v_cond - v_neg)

            if i < N:
                v_keep = (x_t - z_src) / sigma
                v_diff = v_keep - v_pred
                S = v_keep.abs() / (v_keep.abs() + v_diff.abs() + 1e-8)
                mask_high = S >= tau
                v_final = torch.where(mask_high, v_keep, (1 - alpha) * v_keep + alpha * v_pred)
            else:
                v_final = v_pred

            x_t = x_t + dt * v_final

        with torch.no_grad():
            img = self.decode(x_t)
        return img