from typing import Tuple, Optional
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from ..base.flux import FluxBase, register_sampler
from ..base.helper import calculate_shift


@register_sampler(name="uniedit")
class FluxUniEditFlow(FluxBase):
    """
    UniEdit-Flow for FLUX (arXiv:2504.13109): Uni-Inv predictor-corrector inversion
    + Uni-Edit region-adaptive velocity fusion.
    Total NFE = 3·α·N + 1  (e.g. N=15, α=0.6 → 28 NFE).
    """

    @staticmethod
    def _mask(v_diff: torch.Tensor) -> torch.Tensor:
        """MinMaxNorm(ChannelMean(|v_diff|)) → sequence mask in [0,1], shape (B,L,1)."""
        x = v_diff.abs().mean(dim=-1, keepdim=True)   # (B, L, 1)
        min_v = x.amin(dim=1, keepdim=True)            # (B, 1, 1)
        max_v = x.amax(dim=1, keepdim=True)
        return (x - min_v) / (max_v - min_v + 1e-8)

    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               N: int = 15,
               alpha: float = 0.6,
               omega: float = 5.0,
               cfg_scale: float = 3.5,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool, src_txt = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool, tgt_txt = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            z_src, img_ids, h_lat, w_lat = self.prepare_latents(src_img)

        device = self.device if not self.offload else "cuda"

        sigmas = np.linspace(1.0, 1 / N, N)
        mu = calculate_shift(
            z_src.shape[1],
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler, N, device, timesteps=None, sigmas=sigmas, mu=mu
        )
        ts_full = torch.cat([timesteps, torch.zeros(1, device=device, dtype=timesteps.dtype)])

        N_inv = round(alpha * N)
        ts_edit = ts_full[N - N_inv:]
        ts_inv  = ts_edit.flip(0)

        g_src = torch.ones(z_src.shape[0], device=device, dtype=self.dtype)
        g_tgt = torch.full((z_src.shape[0],), cfg_scale, device=device, dtype=self.dtype)

        # Phase 1: Uni-Inv
        x = z_src.clone()
        t0 = ts_inv[0].view(1)
        with torch.no_grad():
            v_hat = self.predict_vector(x, t0, src_emb, src_pool, src_txt, img_ids, g_src)

        pbar = tqdm(range(1, len(ts_inv)), total=N_inv, desc="FLUX Uni-Inv")
        for i in pbar:
            s_prev, s_curr = ts_inv[i - 1], ts_inv[i]
            dt = s_curr - s_prev
            t_curr = s_curr.view(1)

            x_corr = x + dt * v_hat
            with torch.no_grad():
                v_hat = self.predict_vector(x_corr, t_curr, src_emb, src_pool, src_txt, img_ids, g_src)
            x = x + dt * v_hat

        z_inv = x

        # Phase 2: Uni-Edit
        x = z_inv.clone()

        pbar = tqdm(range(len(ts_edit) - 1), total=N_inv, desc="FLUX Uni-Edit")
        for i in pbar:
            s_curr, s_next = ts_edit[i], ts_edit[i + 1]
            dt = s_next - s_curr
            t_curr = s_curr.view(1)

            with torch.no_grad():
                v_src = self.predict_vector(x, t_curr, src_emb, src_pool, src_txt, img_ids, g_src)
                v_tgt = self.predict_vector(x, t_curr, tgt_emb, tgt_pool, tgt_txt, img_ids, g_tgt)

            v_diff = v_tgt - v_src
            m = self._mask(v_diff)

            s = omega * dt * (1 + m) * v_diff
            x = x + s

            v_fused = m * v_tgt + (1 - m) * v_src
            x = x + dt * v_fused

        with torch.no_grad():
            img = self.decode(x, h_lat, w_lat)
        return img
