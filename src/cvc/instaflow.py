from typing import Optional
import torch
from tqdm.auto import tqdm
from ..base.instaflow import InstaFlowBase, register_sampler


@register_sampler(name="cvc")
class InstaFlowCVC(InstaFlowBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               NFE: int = 25,
               alpha: float = 1.0,
               beta: float = 7.0,
               eta: float = 0.2,
               src_prompt_emb: Optional[torch.Tensor] = None,
               tgt_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = self.device if not self.offload else "cuda"

        with torch.no_grad():
            src_emb = src_prompt_emb if src_prompt_emb is not None else self.encode_prompt(src_prompt)
            tgt_emb = tgt_prompt_emb if tgt_prompt_emb is not None else self.encode_prompt(tgt_prompt)
            z_src = self.encode(src_img)

        # Batch [v1 @ qt/src, v2 @ pt/src, v3 @ pt/tgt] in one forward pass
        combined_emb = torch.cat([src_emb, src_emb, tgt_emb], dim=0)

        timesteps = torch.linspace(1.0, 0.0, NFE + 1)[:-1].to(device)

        z_edit = z_src.clone()

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="InstaFlow CVC")
        for i, t in pbar:
            t_curr = t
            t_next = timesteps[i + 1] if i + 1 < NFE else torch.tensor(0.0, device=device)
            dt = t_next - t_curr

            eps = torch.randn_like(z_src)
            qt = (1 - t_curr) * z_src + t_curr * eps
            pt = z_edit + qt - z_src

            with torch.no_grad():
                v_all = self.predict_vector(torch.cat([qt, pt, pt], dim=0), t_curr, combined_emb)
            v1, v2, v3 = v_all.chunk(3, dim=0)

            v_delta = alpha * (v2 - v1) + beta * (v3 - v2)

            dx = v_delta * dt
            grad = 2.0 * dt * (dx - z_src)
            v_new = v_delta - eta * grad

            z_edit = z_edit + eta * dt * v_new

        with torch.no_grad():
            img = self.decode(z_edit)
        return img