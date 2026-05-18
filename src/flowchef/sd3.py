from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.sd3 import StableDiffusion3Base, register_sampler


@register_sampler(name="flowchef")
class SD3FlowChef(StableDiffusion3Base):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 30,
               guidance_scale: float = 10.0,
               cfg_scale: float = 7.5,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None,
               neg_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        """
        FlowChef for SD3 (arXiv:2412.00100): steering rectified flow via
        loss-based gradient guidance with gradient skipping.
        Starts from pure noise — no inversion.
        """
        with torch.no_grad():
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            neg_emb, neg_pool = self.prepare_embed(neg_prompt, neg_prompt_emb)
            z_src = self.encode(src_img)

        x_t = torch.randn_like(z_src)

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        pbar = tqdm(timesteps, total=NFE, desc="SD3 FlowChef")
        for i, t in enumerate(pbar):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else torch.tensor(0.0).to(self.device)
            dt = sigma_next - sigma

            with torch.no_grad():
                v_cond = self.predict_vector(x_t, t, tgt_emb, tgt_pool)
                v_neg = self.predict_vector(x_t, t, neg_emb, neg_pool)
                v = v_neg + cfg_scale * (v_cond - v_neg)

            x0_hat = x_t - sigma * v
            loss_grad = x0_hat - z_src
            x_t = x_t + dt * v - guidance_scale * loss_grad

        with torch.no_grad():
            img = self.decode(x_t)
        return img
