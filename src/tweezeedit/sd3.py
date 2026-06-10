from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.sd3 import StableDiffusion3Base, register_sampler


@register_sampler(name="tweezeedit")
class SD3TweezeEdit(StableDiffusion3Base):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 50,
               m: Optional[int] = None,
               gamma: float = 1.0,
               tgt_cfg_scale: float = 13.5,
               src_cfg_scale: float = 3.5,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None,
               neg_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        if m is None:
            m = NFE // 2

        with torch.no_grad():
            src_emb, src_pool = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            neg_emb, neg_pool = self.prepare_embed(neg_prompt, neg_prompt_emb)
            z_src = self.encode(src_img)

        x_t = z_src.clone()

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="SD3 TweezeEdit")
        for i, t in pbar:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = sigma_next - sigma  # negative

            eps = torch.randn_like(z_src)
            qt = (1 - sigma) * z_src + sigma * eps
            pt = x_t + qt - z_src

            with torch.no_grad():
                v_src_neg  = self.predict_vector(qt, t, neg_emb, neg_pool)
                v_src_cond = self.predict_vector(qt, t, src_emb, src_pool)
                v_src = v_src_neg + src_cfg_scale * (v_src_cond - v_src_neg)

                v_tar_neg  = self.predict_vector(pt, t, neg_emb, neg_pool)
                v_tar_cond = self.predict_vector(pt, t, tgt_emb, tgt_pool)
                v_tar = v_tar_neg + tgt_cfg_scale * (v_tar_cond - v_tar_neg)

            z0_hat_src = qt - sigma * v_src
            z0_hat_tar = pt - sigma * v_tar

            gamma_hat = -gamma * dt.abs()
            grad = gamma_hat * (z0_hat_src - z_src) if i < m else torch.zeros_like(z_src)

            x_t = z_src + (1 - sigma_next) * (z0_hat_tar - z0_hat_src) - grad

        with torch.no_grad():
            img = self.decode(x_t)
        return img