from typing import Optional
import torch
from tqdm.auto import tqdm
from ..base.instaflow import InstaFlowBase, register_sampler


@register_sampler(name="veloedit")
class InstaFlowVeloEdit(InstaFlowBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 25,
               N: int = 1,
               tau: float = 0.4,
               alpha: float = 0.8,
               cfg_scale: float = 16.5,
               src_prompt_emb: Optional[torch.Tensor] = None,
               tgt_prompt_emb: Optional[torch.Tensor] = None,
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            tgt_emb, tgt_neg = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            combined_embeds = torch.cat([tgt_neg, tgt_emb], dim=0)
            z_src = self.encode(src_img)

        x_t = torch.randn_like(z_src)

        timesteps = torch.linspace(1.0, 0.0, NFE + 1)[:-1].to(self.device)

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="InstaFlow VeloEdit")
        for i, t in pbar:
            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = t_next - t_curr

            with torch.no_grad():
                latent_input = torch.cat([x_t, x_t])
                v_all = self.predict_vector(latent_input, t_curr, combined_embeds)
                v_neg, v_cond = v_all.chunk(2)
                v_pred = v_neg + cfg_scale * (v_cond - v_neg)

            if i < N:
                v_keep = (x_t - z_src) / t_curr
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