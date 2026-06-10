from typing import Tuple, Optional
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from ..base.flux import FluxBase, register_sampler
from ..base.helper import calculate_shift


@register_sampler(name="fireflow")
class FluxFireFlow(FluxBase):
    _INJ_START = 20
    _INJ_END   = 37

    def _make_v_hooks(self, store: dict, mode: str, is_midpoint: bool) -> list:
        handles = []
        for idx in range(self._INJ_START, self._INJ_END + 1):
            block = self.transformer.single_transformer_blocks[idx]
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

    def _predict(self, z, t_tensor, emb, pool, txt_ids, img_ids, guidance,
                 store=None, mode=None, is_midpoint=False):
        if store is not None and mode is not None:
            handles = self._make_v_hooks(store, mode, is_midpoint)
            try:
                with torch.no_grad():
                    v = self.predict_vector(z, t_tensor, emb, pool, txt_ids, img_ids, guidance=guidance)
            finally:
                for h in handles:
                    h.remove()
            return v
        with torch.no_grad():
            return self.predict_vector(z, t_tensor, emb, pool, txt_ids, img_ids, guidance=guidance)

    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               NFE: int = 8,
               cfg_scale: float = 5.0,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool, src_txt = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool, tgt_txt = self.prepare_embed(tgt_prompt, tgt_prompt_emb)
            z_src, img_ids, h_lat, w_lat = self.prepare_latents(src_img)

        device = self.device if not self.offload else "cuda"

        sigmas = np.linspace(1.0, 1 / NFE, NFE)
        mu = calculate_shift(
            z_src.shape[1],
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler, NFE, device, timesteps=None, sigmas=sigmas, mu=mu
        )
        ts_den = torch.cat([timesteps, torch.zeros(1, device=device, dtype=timesteps.dtype)])
        ts_inv = ts_den.flip(0)
        N_steps = len(ts_den) - 1

        g_src = torch.ones(z_src.shape[0], device=device, dtype=self.dtype)
        g_tgt = torch.full((z_src.shape[0],), cfg_scale, device=device, dtype=self.dtype)

        v_store: dict = {}

        # Phase 1: Inversion
        x = z_src.clone()
        hat_v = None

        pbar = tqdm(enumerate(zip(ts_inv[:-1], ts_inv[1:])), total=N_steps, desc="FLUX FireFlow Inversion")
        for i, (s_c, s_n) in pbar:
            is_last = (i == N_steps - 1)
            dt = s_n - s_c
            s_m = s_c + dt / 2
            t_c = s_c.view(1)
            t_m = s_m.view(1)

            if hat_v is None:
                v = self._predict(x, t_c, src_emb, src_pool, src_txt, img_ids, g_src,
                                  store=v_store if is_last else None, mode='save', is_midpoint=False)
            else:
                v = hat_v

            x_mid = x + (dt / 2) * v
            v_mid = self._predict(x_mid, t_m, src_emb, src_pool, src_txt, img_ids, g_src,
                                  store=v_store if is_last else None, mode='save', is_midpoint=True)
            hat_v = v_mid
            x = x + dt * v_mid

        x_noise = x

        # Phase 2: Denoising -- inject stored src V-features at every step
        x = x_noise
        hat_v = None

        pbar = tqdm(zip(ts_den[:-1], ts_den[1:]), total=N_steps, desc="FLUX FireFlow Denoising")
        for s_c, s_n in pbar:
            dt = s_n - s_c
            s_m = s_c + dt / 2
            t_c = s_c.view(1)
            t_m = s_m.view(1)

            if hat_v is None:
                v = self._predict(x, t_c, tgt_emb, tgt_pool, tgt_txt, img_ids, g_tgt,
                                  store=v_store or None,
                                  mode='inject', is_midpoint=False)
            else:
                v = hat_v

            x_mid = x + (dt / 2) * v
            v_mid = self._predict(x_mid, t_m, tgt_emb, tgt_pool, tgt_txt, img_ids, g_tgt,
                                  store=v_store or None,
                                  mode='inject', is_midpoint=True)
            hat_v = v_mid
            x = x + dt * v_mid

        with torch.no_grad():
            img = self.decode(x, h_lat, w_lat)
        return img