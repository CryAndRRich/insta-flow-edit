from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.sd3 import StableDiffusion3Base, register_sampler


@register_sampler(name="uniedit")
class SD3UniEditFlow(StableDiffusion3Base):
    """
    UniEdit-Flow for SD3 (arXiv:2504.13109): predictor-corrector inversion (Uni-Inv)
    + region-adaptive velocity fusion editing (Uni-Edit).
    Total NFE = 3·α·N + 1  (e.g. N=15, α=0.6 → 28 NFE).
    """

    @staticmethod
    def _mask(v_diff: torch.Tensor) -> torch.Tensor:
        """MinMaxNorm(ChannelMean(|v_diff|)) → spatial mask in [0,1], shape (B,1,H,W)."""
        x = v_diff.abs().mean(dim=1, keepdim=True)
        min_v = x.amin(dim=(-2, -1), keepdim=True)
        max_v = x.amax(dim=(-2, -1), keepdim=True)
        return (x - min_v) / (max_v - min_v + 1e-8)

    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               N: int = 15,
               alpha: float = 0.6,
               omega: float = 5.0,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None,
               neg_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            z_src = self.encode(src_img)

        device = self.device
        scale = self.scheduler.config.num_train_timesteps

        self.scheduler.set_timesteps(N, device=device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / scale
        ts_full = torch.cat([sigmas, torch.zeros(1, device=device, dtype=sigmas.dtype)])

        N_inv = round(alpha * N)
        ts_edit = ts_full[N - N_inv:]
        ts_inv  = ts_edit.flip(0)

        def to_t(sigma):
            return (sigma * scale).to(dtype=self.dtype)

        # Phase 1: Uni-Inv
        x = z_src.clone()

        t0 = to_t(ts_inv[0]).expand(x.shape[0])
        with torch.no_grad():
            v_hat = self.predict_vector(x, t0, src_emb, src_pool)

        pbar = tqdm(range(1, len(ts_inv)), total=N_inv, desc="SD3 Uni-Inv")
        for i in pbar:
            sigma_prev, sigma_curr = ts_inv[i - 1], ts_inv[i]
            dt = sigma_curr - sigma_prev
            t_curr = to_t(sigma_curr).expand(x.shape[0])

            x_corr = x + dt * v_hat
            with torch.no_grad():
                v_hat = self.predict_vector(x_corr, t_curr, src_emb, src_pool)
            x = x + dt * v_hat

        z_inv = x

        # Phase 2: Uni-Edit
        x = z_inv.clone()

        pbar = tqdm(range(len(ts_edit) - 1), total=N_inv, desc="SD3 Uni-Edit")
        for i in pbar:
            sigma_curr, sigma_next = ts_edit[i], ts_edit[i + 1]
            dt = sigma_next - sigma_curr
            t_curr = to_t(sigma_curr).expand(x.shape[0])

            with torch.no_grad():
                v_src = self.predict_vector(x, t_curr, src_emb, src_pool)
                v_tgt = self.predict_vector(x, t_curr, tgt_emb, tgt_pool)

            v_diff = v_tgt - v_src
            m = self._mask(v_diff)

            s = omega * dt * (1 + m) * v_diff
            x = x + s

            v_fused = m * v_tgt + (1 - m) * v_src
            x = x + dt * v_fused

        with torch.no_grad():
            img = self.decode(x)
        return img
