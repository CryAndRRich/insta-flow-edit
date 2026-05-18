from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.instaflow import InstaFlowBase, register_sampler


@register_sampler(name="uniedit")
class InstaFlowUniEdit(InstaFlowBase):
    """
    UniEdit-Flow for InstaFlow (arXiv:2504.13109): Uni-Inv predictor-corrector
    inversion + Uni-Edit region-adaptive velocity fusion.
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
               src_prompt_emb: Optional[torch.Tensor] = None,
               tgt_prompt_emb: Optional[torch.Tensor] = None,
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, _ = self.prepare_embed(src_prompt, "", src_prompt_emb)
            tgt_emb, _ = self.prepare_embed(tgt_prompt, "", tgt_prompt_emb)
            combined_emb = torch.cat([src_emb, tgt_emb], dim=0)
            z_src = self.encode(src_img)

        ts_full = torch.linspace(1.0, 0.0, N + 1).to(self.device)
        N_inv   = round(alpha * N)
        ts_edit = ts_full[N - N_inv:]
        ts_inv  = ts_edit.flip(0)

        # Phase 1: Uni-Inv
        x = z_src.clone()

        with torch.no_grad():
            v_hat = self.predict_vector(x, ts_inv[0], src_emb)

        pbar = tqdm(range(1, len(ts_inv)), total=N_inv, desc="InstaFlow Uni-Inv")
        for i in pbar:
            t_prev, t_curr = ts_inv[i - 1], ts_inv[i]
            dt = t_curr - t_prev

            x_corr = x + dt * v_hat
            with torch.no_grad():
                v_hat = self.predict_vector(x_corr, t_curr, src_emb)
            x = x + dt * v_hat

        z_inv = x

        # Phase 2: Uni-Edit
        x = z_inv.clone()

        pbar = tqdm(range(len(ts_edit) - 1), total=N_inv, desc="InstaFlow Uni-Edit")
        for i in pbar:
            t_curr = ts_edit[i]
            t_next = ts_edit[i + 1]
            dt = t_next - t_curr

            with torch.no_grad():
                v_all = self.predict_vector(torch.cat([x, x], dim=0), t_curr, combined_emb)
                v_src, v_tgt = v_all.chunk(2)

            v_diff = v_tgt - v_src
            m = self._mask(v_diff)

            s = omega * dt * (1 + m) * v_diff
            x = x + s

            v_fused = m * v_tgt + (1 - m) * v_src
            x = x + dt * v_fused

        with torch.no_grad():
            img = self.decode(x)
        return img
