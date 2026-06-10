from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.instaflow import InstaFlowBase, register_sampler


@register_sampler(name="fireflow")
class InstaFlowFireFlow(InstaFlowBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 25,
               cfg_scale: float = 7.5,
               src_prompt_emb: Optional[torch.Tensor] = None,
               tgt_prompt_emb: Optional[torch.Tensor] = None,
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, _ = self.prepare_embed(src_prompt, "", src_prompt_emb, None)
            tgt_emb, tgt_neg = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            den_embeds = torch.cat([tgt_neg, tgt_emb], dim=0)
            z_src = self.encode(src_img)

        ts_inv = torch.linspace(0.0, 1.0, NFE + 1).to(self.device)
        ts_den = ts_inv.flip(0)
        N_steps = NFE

        # Phase 1: Inversion
        x = z_src.clone()
        hat_v = None

        pbar = tqdm(zip(ts_inv[:-1], ts_inv[1:]), total=N_steps, desc="InstaFlow FireFlow Inversion")
        for s_c, s_n in pbar:
            dt = s_n - s_c
            s_m = s_c + dt / 2

            with torch.no_grad():
                if hat_v is None:
                    v = self.predict_vector(x, s_c, src_emb)
                else:
                    v = hat_v

                x_mid = x + (dt / 2) * v
                v_mid = self.predict_vector(x_mid, s_m, src_emb)
                hat_v = v_mid
                x = x + dt * v_mid

        x_noise = x

        # Phase 2: Denoising
        x = x_noise
        hat_v = None

        pbar = tqdm(zip(ts_den[:-1], ts_den[1:]), total=N_steps, desc="InstaFlow FireFlow Denoising")
        for s_c, s_n in pbar:
            dt = s_n - s_c
            s_m = s_c + dt / 2

            with torch.no_grad():
                if hat_v is None:
                    v_all = self.predict_vector(torch.cat([x, x]), s_c, den_embeds)
                    v_neg, v_cond = v_all.chunk(2)
                    v = v_neg + cfg_scale * (v_cond - v_neg)
                else:
                    v = hat_v

                x_mid = x + (dt / 2) * v

                v_mid_all = self.predict_vector(torch.cat([x_mid, x_mid]), s_m, den_embeds)
                v_mid_neg, v_mid_cond = v_mid_all.chunk(2)
                v_mid = v_mid_neg + cfg_scale * (v_mid_cond - v_mid_neg)
                hat_v = v_mid
                x = x + dt * v_mid

        with torch.no_grad():
            img = self.decode(x)
        return img