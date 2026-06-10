from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.sd3 import StableDiffusion3Base, register_sampler


@register_sampler(name="cvc")
class SD3CVC(StableDiffusion3Base):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               NFE: int = 50,
               alpha: float = 1.0,
               beta: float = 7.0,
               eta: float = 0.2,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            z_src = self.encode(src_img)

        self.scheduler.set_timesteps(NFE, device=self.device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / self.scheduler.config.num_train_timesteps

        z_edit = z_src.clone()

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="SD3 CVC")
        for i, t in pbar:
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1] if i + 1 < NFE else torch.tensor(0.0, device=self.device)
            dt = sigma_next - sigma

            eps = torch.randn_like(z_src)
            qt = (1 - sigma) * z_src + sigma * eps
            pt = z_edit + qt - z_src

            with torch.no_grad():
                v1 = self.predict_vector(qt, t, src_emb, src_pool)
                v2 = self.predict_vector(pt, t, src_emb, src_pool)
                v3 = self.predict_vector(pt, t, tgt_emb, tgt_pool)

            v_delta = alpha * (v2 - v1) + beta * (v3 - v2)

            dx = v_delta * dt
            grad = 2.0 * dt * (dx - z_src)
            v_new = v_delta - eta * grad

            z_edit = z_edit + eta * dt * v_new

        with torch.no_grad():
            img = self.decode(z_edit)
        return img