from typing import Tuple, Optional
import torch
from ..base.flux import FluxBase, register_sampler


@register_sampler(name="chordedit")
class FluxChordEdit(FluxBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               sigma_t: float = 0.90,
               delta: float = 0.15,
               lambda_: float = 1.0,
               sigma_c: float = 0.30,
               cfg_scale: float = 3.5,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool, src_txt_ids = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool, tgt_txt_ids = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            z_src, img_ids, h_lat, w_lat = self.prepare_latents(src_img)

        device = self.device if not self.offload else "cuda"
        guidance = torch.tensor([cfg_scale], device=device).expand(z_src.shape[0])

        t  = torch.tensor(sigma_t,         device=device, dtype=self.dtype)
        td = torch.tensor(sigma_t - delta, device=device, dtype=self.dtype)
        tc = torch.tensor(sigma_c,         device=device, dtype=self.dtype)

        with torch.no_grad():
            # R at sigma_t: DeltaQ = Q_tar - Q_src (neg cancels in difference)
            eps1 = torch.randn_like(z_src)
            z_t = (1 - t) * z_src + t * eps1
            v_src_t = self.predict_vector(z_t, t.view(1), src_emb, src_pool, src_txt_ids, img_ids, guidance=guidance)
            v_tar_t = self.predict_vector(z_t, t.view(1), tgt_emb, tgt_pool, tgt_txt_ids, img_ids, guidance=guidance)
            R_t = v_tar_t - v_src_t

            # R at sigma_t - delta
            eps2 = torch.randn_like(z_src)
            z_td = (1 - td) * z_src + td * eps2
            v_src_td = self.predict_vector(z_td, td.view(1), src_emb, src_pool, src_txt_ids, img_ids, guidance=guidance)
            v_tar_td = self.predict_vector(z_td, td.view(1), tgt_emb, tgt_pool, tgt_txt_ids, img_ids, guidance=guidance)
            R_td = v_tar_td - v_src_td

            # Chord control field and single Euler step
            u_hat = (t * R_td + delta * R_t) / (t + delta)
            x_pred = z_src + lambda_ * u_hat

            # Proximal refinement: x_0 = x_pred - sigma_c * v(x_pred, sigma_c, c_tar)
            v_prox = self.predict_vector(x_pred, tc.view(1), tgt_emb, tgt_pool, tgt_txt_ids, img_ids, guidance=guidance)
            x_tar = x_pred - tc * v_prox

            img = self.decode(x_tar, h_lat, w_lat)
        return img