from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.instaflow import InstaFlowBase, register_sampler


@register_sampler(name="flowchef")
class InstaFlowChef(InstaFlowBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 25,
               guidance_scale: float = 15.0,
               cfg_scale: float = 10.0,
               src_prompt_emb: Optional[torch.Tensor] = None,
               tgt_prompt_emb: Optional[torch.Tensor] = None,
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        FlowChef for InstaFlow (arXiv:2412.00100): steering rectified flow via
        loss-based gradient guidance with gradient skipping.
        Starts from pure noise — no inversion.
        """
        with torch.no_grad():
            tgt_emb, tgt_neg = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            combined_embeds = torch.cat([tgt_neg, tgt_emb], dim=0)
            z_src = self.encode(src_img)

        x_t = torch.randn_like(z_src)

        timesteps = torch.linspace(1.0, 0.0, NFE + 1)[:-1].to(self.device)

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="InstaFlow FlowChef")
        for i, t in pbar:
            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = t_next - t_curr

            with torch.no_grad():
                latent_input = torch.cat([x_t, x_t])
                v_all = self.predict_vector(latent_input, t_curr, combined_embeds)
                v_neg, v_cond = v_all.chunk(2)
                v = v_neg + cfg_scale * (v_cond - v_neg)

            x0_hat = x_t - t_curr * v
            loss_grad = x0_hat - z_src
            x_t = x_t + dt * v - guidance_scale * loss_grad

        with torch.no_grad():
            img = self.decode(x_t)
        return img
