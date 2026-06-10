from typing import Optional
import torch
from ..base.instaflow import InstaFlowBase, register_sampler


@register_sampler(name="chordedit")
class InstaFlowChordEdit(InstaFlowBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               sigma_t: float = 0.90,
               delta: float = 0.15,
               lambda_: float = 1.0,
               sigma_c: float = 0.30,
               cfg_scale: float = 16.5,
               src_prompt_emb: Optional[torch.Tensor] = None,
               tgt_prompt_emb: Optional[torch.Tensor] = None,
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_neg = self.prepare_embed(src_prompt, neg_prompt, src_prompt_emb, neg_prompt_emb)
            tgt_emb, tgt_neg = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            z_src = self.encode(src_img)

        combined = torch.cat([src_neg, src_emb, tgt_neg, tgt_emb], dim=0)
        t  = torch.tensor(sigma_t,         device=self.device)
        td = torch.tensor(sigma_t - delta, device=self.device)
        tc = torch.tensor(sigma_c,         device=self.device)

        with torch.no_grad():
            # R_t: batch [z_t]*4 with [src_neg, src, tgt_neg, tgt]
            eps1 = torch.randn_like(z_src)
            z_t = (1 - t) * z_src + t * eps1
            v_t = self.predict_vector(z_t.expand(4, -1, -1, -1), t, combined)
            v_t_src_neg, v_t_src_cond, v_t_tar_neg, v_t_tar_cond = v_t.chunk(4)
            Q_src_t = v_t_src_neg + cfg_scale * (v_t_src_cond - v_t_src_neg)
            Q_tar_t = v_t_tar_neg + cfg_scale * (v_t_tar_cond - v_t_tar_neg)
            R_t = Q_tar_t - Q_src_t

            # R_td: batch [z_td]*4 with [src_neg, src, tgt_neg, tgt]
            eps2 = torch.randn_like(z_src)
            z_td = (1 - td) * z_src + td * eps2
            v_td = self.predict_vector(z_td.expand(4, -1, -1, -1), td, combined)
            v_td_src_neg, v_td_src_cond, v_td_tar_neg, v_td_tar_cond = v_td.chunk(4)
            Q_src_td = v_td_src_neg + cfg_scale * (v_td_src_cond - v_td_src_neg)
            Q_tar_td = v_td_tar_neg + cfg_scale * (v_td_tar_cond - v_td_tar_neg)
            R_td = Q_tar_td - Q_src_td

            # Chord control field and single Euler step
            u_hat = (t * R_td + delta * R_t) / (t + delta)
            x_pred = z_src + lambda_ * u_hat

            # Proximal refinement: x_0 prediction from x_pred at noise level sigma_c
            v_prox = self.predict_vector(x_pred.expand(2, -1, -1, -1), tc,
                                         torch.cat([tgt_neg, tgt_emb], dim=0))
            v_prox_neg, v_prox_cond = v_prox.chunk(2)
            v_prox_cfg = v_prox_neg + cfg_scale * (v_prox_cond - v_prox_neg)
            x_tar = x_pred - tc * v_prox_cfg

            img = self.decode(x_tar)
        return img