from typing import Optional
import torch
from tqdm.auto import tqdm
from ..base.instaflow import InstaFlowBase, register_sampler


@register_sampler(name="flowslider")
class InstaFlowSlider(InstaFlowBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 25,
               n_start: int = 0,
               strength: float = 1.0,
               cfg_scale: float = 16.5,
               src_prompt_emb: Optional[torch.Tensor] = None,
               tgt_prompt_emb: Optional[torch.Tensor] = None,
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_neg = self.prepare_embed(src_prompt, neg_prompt, src_prompt_emb, neg_prompt_emb)
            tgt_emb, tgt_neg = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            # For qt: [neg, src]; for pt: [neg, src, tgt] -- embed neg once (same neg_prompt)
            z_src = self.encode(src_img)

        x_t = z_src.clone()

        timesteps = torch.linspace(1.0, 0.0, NFE + 1)[:-1].to(self.device)

        pbar = tqdm(enumerate(timesteps), total=NFE, desc="InstaFlow FlowSlider")
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
                # Pass 1: V(qt, c_src) via batch [neg@qt, src@qt]
                v_qt_all = self.predict_vector(torch.cat([qt, qt]), t_curr,
                                               torch.cat([src_neg, src_emb]))
                v_qt_neg, v_qt_src_cond = v_qt_all.chunk(2)
                v_qt_src = v_qt_neg + cfg_scale * (v_qt_src_cond - v_qt_neg)

                # Pass 2: V(pt, c_src) and V(pt, c_tar) via batch [neg@pt, src@pt, tgt@pt]
                v_pt_all = self.predict_vector(torch.cat([pt, pt, pt]), t_curr,
                                               torch.cat([src_neg, src_emb, tgt_emb]))
                v_pt_neg, v_pt_src_cond, v_pt_tar_cond = v_pt_all.chunk(3)
                v_pt_src = v_pt_neg + cfg_scale * (v_pt_src_cond - v_pt_neg)
                v_pt_tar = v_pt_neg + cfg_scale * (v_pt_tar_cond - v_pt_neg)

            V_steer = v_pt_tar - v_pt_src
            V_fid   = v_pt_src - v_qt_src
            x_t = x_t + dt * (V_fid + strength * V_steer)

        with torch.no_grad():
            img = self.decode(x_t)
        return img