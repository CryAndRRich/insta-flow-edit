from typing import Tuple, Optional
import torch
from ..base.sd3 import StableDiffusion3Base, register_sampler


@register_sampler(name="chordedit")
class SD3ChordEdit(StableDiffusion3Base):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               sigma_t: float = 0.90,
               delta: float = 0.15,
               lambda_: float = 1.0,
               sigma_c: float = 0.30,
               cfg_scale: float = 7.5,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None,
               neg_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            neg_emb, neg_pool = self.prepare_embed(neg_prompt, neg_prompt_emb)
            z_src = self.encode(src_img)

        num_train_ts = self.scheduler.config.num_train_timesteps
        sigma_a = torch.tensor(sigma_t,         device=self.device, dtype=self.dtype)
        sigma_d = torch.tensor(sigma_t - delta, device=self.device, dtype=self.dtype)
        sigma_cv = torch.tensor(sigma_c,        device=self.device, dtype=self.dtype)
        t_a = (sigma_a * num_train_ts).view(1)
        t_d = (sigma_d * num_train_ts).view(1)
        t_c = (sigma_cv * num_train_ts).view(1)

        with torch.no_grad():
            # R_t: neg cancels, so R = cfg_scale * (v_tar_cond - v_src_cond)
            eps1 = torch.randn_like(z_src)
            z_t = (1 - sigma_a) * z_src + sigma_a * eps1
            v_src_t = self.predict_vector(z_t, t_a, src_emb, src_pool)
            v_tar_t = self.predict_vector(z_t, t_a, tgt_emb, tgt_pool)
            R_t = cfg_scale * (v_tar_t - v_src_t)

            # R_td
            eps2 = torch.randn_like(z_src)
            z_td = (1 - sigma_d) * z_src + sigma_d * eps2
            v_src_td = self.predict_vector(z_td, t_d, src_emb, src_pool)
            v_tar_td = self.predict_vector(z_td, t_d, tgt_emb, tgt_pool)
            R_td = cfg_scale * (v_tar_td - v_src_td)

            # Chord control field and single Euler step
            u_hat = (sigma_a * R_td + delta * R_t) / (sigma_a + delta)
            x_pred = z_src + lambda_ * u_hat

            # Proximal refinement with full CFG
            v_prox_neg  = self.predict_vector(x_pred, t_c, neg_emb, neg_pool)
            v_prox_cond = self.predict_vector(x_pred, t_c, tgt_emb, tgt_pool)
            v_prox = v_prox_neg + cfg_scale * (v_prox_cond - v_prox_neg)
            x_tar = x_pred - sigma_cv * v_prox

            img = self.decode(x_tar)
        return img