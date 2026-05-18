from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.instaflow import InstaFlowBase, register_sampler


@register_sampler(name="flowalign")
class InstaFlowAlign(InstaFlowBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 25,
               n_start: int = 0,
               cfg_scale: float = 16.5,
               zeta: float = 0.01,
               src_prompt_emb: Optional[torch.Tensor] = None,
               tgt_prompt_emb: Optional[torch.Tensor] = None,
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, _ = self.prepare_embed(src_prompt, neg_prompt, src_prompt_emb, neg_prompt_emb)
            tgt_emb, _ = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            combined_embeds = torch.cat([src_emb, src_emb, tgt_emb], dim=0)

        with torch.no_grad():
            z_src = self.encode(src_img)

        x_t = z_src.clone()

        timesteps = torch.linspace(1.0, 0.0, NFE + 1)[:-1].to(self.device)

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="InstaFlow FlowAlign")
        for i, t in pbar:
            if i < n_start:
                continue

            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = t_next - t_curr

            eps = torch.randn_like(z_src)
            qt = (1 - t_curr) * z_src + t_curr * eps
            pt = x_t + qt - z_src

            with torch.no_grad():
                latent_input = torch.cat([qt, pt, pt])
                v_all = self.predict_vector(latent_input, t_curr, combined_embeds)
                v_src, v_tar_src, v_tar_text = v_all.chunk(3)
                v_tar = v_tar_src + cfg_scale * (v_tar_text - v_tar_src)

            reg_term = (qt - pt) + t_curr * (v_tar - v_src)
            x_t = x_t + dt * (v_tar - v_src) + zeta * reg_term

        with torch.no_grad():
            img = self.decode(x_t)
        return img
