from typing import Tuple, Optional
import torch
from tqdm.auto import tqdm
from ..base.sd3 import StableDiffusion3Base, register_sampler


@register_sampler(name="fireflow")
class SD3FireFlow(StableDiffusion3Base):
    """
    FireFlow for SD3 (arXiv:2412.07517): modified midpoint ODE inversion + editing
    with self-attention Value-feature injection into the last 12 transformer blocks.
    """

    _INJ_START = 12
    _INJ_END   = 23

    def _make_v_hooks(self, store: dict, mode: str, is_midpoint: bool) -> list:
        handles = []
        for idx in range(self._INJ_START, self._INJ_END + 1):
            block = self.transformer.transformer_blocks[idx]
            key = (idx, is_midpoint)
            if mode == 'save':
                def _hook(module, inp, out, k=key):
                    store[k] = out.detach().cpu()
                h = block.attn.to_v.register_forward_hook(_hook)
            else:
                def _hook(module, inp, out, k=key):
                    if k in store:
                        return store[k].to(out.device, dtype=out.dtype)
                h = block.attn.to_v.register_forward_hook(_hook)
            handles.append(h)
        return handles

    def _predict(self, z, t, emb, pool, store=None, mode=None, is_midpoint=False):
        if store is not None and mode is not None:
            handles = self._make_v_hooks(store, mode, is_midpoint)
            try:
                with torch.no_grad():
                    v = self.predict_vector(z, t, emb, pool)
            finally:
                for h in handles:
                    h.remove()
            return v
        with torch.no_grad():
            return self.predict_vector(z, t, emb, pool)

    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               NFE: int = 28,
               cfg_scale: float = 13.5,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None,
               neg_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            neg_emb, neg_pool = self.prepare_embed(neg_prompt, neg_prompt_emb)
            z_src = self.encode(src_img)

        device = self.device
        scale = self.scheduler.config.num_train_timesteps

        self.scheduler.set_timesteps(NFE, device=device)
        timesteps = self.scheduler.timesteps
        sigmas = timesteps / scale

        ts_den = torch.cat([sigmas, torch.zeros(1, device=device, dtype=sigmas.dtype)])
        ts_inv = ts_den.flip(0)
        N_steps = len(ts_den) - 1

        def to_t(sigma):
            return (sigma * scale).to(dtype=self.dtype)

        v_store: dict = {}

        # Phase 1: Inversion
        x = z_src.clone()
        hat_v = None

        pbar = tqdm(enumerate(zip(ts_inv[:-1], ts_inv[1:])), total=N_steps, desc="SD3 FireFlow Inversion")
        for i, (s_c, s_n) in pbar:
            is_last = (i == N_steps - 1)
            dt = s_n - s_c
            s_m = s_c + dt / 2
            t_c = to_t(s_c)
            t_m = to_t(s_m)

            if hat_v is None:
                v = self._predict(x, t_c, src_emb, src_pool,
                                  store=v_store if is_last else None, mode='save', is_midpoint=False)
            else:
                v = hat_v

            x_mid = x + (dt / 2) * v
            v_mid = self._predict(x_mid, t_m, src_emb, src_pool,
                                  store=v_store if is_last else None, mode='save', is_midpoint=True)
            hat_v = v_mid
            x = x + dt * v_mid

        x_noise = x

        # Phase 2: Denoising
        x = x_noise
        hat_v = None

        pbar = tqdm(enumerate(zip(ts_den[:-1], ts_den[1:])), total=N_steps, desc="SD3 FireFlow Denoising")
        for i, (s_c, s_n) in pbar:
            is_first = (i == 0)
            dt = s_n - s_c
            s_m = s_c + dt / 2
            t_c = to_t(s_c)
            t_m = to_t(s_m)

            if hat_v is None:
                use_store = v_store if (is_first and v_store) else None
                v_cond = self._predict(x, t_c, tgt_emb, tgt_pool,
                                       store=use_store, mode='inject', is_midpoint=False)
                v_neg = self._predict(x, t_c, neg_emb, neg_pool)
                v = v_neg + cfg_scale * (v_cond - v_neg)
            else:
                v = hat_v

            x_mid = x + (dt / 2) * v

            use_store = v_store if (is_first and v_store) else None
            v_mid_cond = self._predict(x_mid, t_m, tgt_emb, tgt_pool,
                                       store=use_store, mode='inject', is_midpoint=True)
            v_mid_neg = self._predict(x_mid, t_m, neg_emb, neg_pool)
            v_mid = v_mid_neg + cfg_scale * (v_mid_cond - v_mid_neg)
            hat_v = v_mid
            x = x + dt * v_mid

        with torch.no_grad():
            img = self.decode(x)
        return img
