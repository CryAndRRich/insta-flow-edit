from typing import Optional
import torch
from tqdm.auto import tqdm
from ..base.instaflow import InstaFlowBase, register_sampler


@register_sampler(name="tweezeedit")
class InstaFlowTweezeEdit(InstaFlowBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 25,
               m: Optional[int] = None,
               gamma: float = 1.0,
               tgt_cfg_scale: float = 16.5,
               src_cfg_scale: float = 1.5,
               src_prompt_emb: Optional[torch.Tensor] = None,
               tgt_prompt_emb: Optional[torch.Tensor] = None,
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        if m is None:
            m = NFE // 2

        with torch.no_grad():
            src_emb, src_neg = self.prepare_embed(src_prompt, neg_prompt, src_prompt_emb, neg_prompt_emb)
            tgt_emb, tgt_neg = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            combined_embeds = torch.cat([src_neg, src_emb, tgt_neg, tgt_emb], dim=0)
            z_src = self.encode(src_img)

        x_t = z_src.clone()
        timesteps = torch.linspace(1.0, 0.0, NFE + 1)[:-1].to(self.device)

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="InstaFlow TweezeEdit")
        for i, t in pbar:
            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = t_next - t_curr  # negative

            eps = torch.randn_like(z_src)
            qt = (1 - t_curr) * z_src + t_curr * eps
            pt = x_t + qt - z_src

            with torch.no_grad():
                latent_input = torch.cat([qt, qt, pt, pt])
                v_all = self.predict_vector(latent_input, t_curr, combined_embeds)
                v_src_neg, v_src_cond, v_tar_neg, v_tar_cond = v_all.chunk(4)
                v_src = v_src_neg + src_cfg_scale * (v_src_cond - v_src_neg)
                v_tar = v_tar_neg + tgt_cfg_scale * (v_tar_cond - v_tar_neg)

            z0_hat_src = qt - t_curr * v_src
            z0_hat_tar = pt - t_curr * v_tar

            gamma_hat = -gamma * dt.abs()
            grad = gamma_hat * (z0_hat_src - z_src) if i < m else torch.zeros_like(z_src)

            x_t = z_src + (1 - t_next) * (z0_hat_tar - z0_hat_src) - grad

        with torch.no_grad():
            img = self.decode(x_t)
        return img