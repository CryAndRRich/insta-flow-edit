from typing import Tuple, Optional
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from ..base.flux import FluxBase, register_sampler
from ..base.helper import calculate_shift, lr_hump_tail_beta


@register_sampler(name="dvrf")
class FluxDeltaVelFlow(FluxBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               steps: int = 50,
               eta: float = 1.0,
               lr_max: float = 0.04,
               tar_cfg_scale: float = 5.5,
               src_cfg_scale: float = 1.0,
               src_prompt_emb: Optional[Tuple] = None,
               tgt_prompt_emb: Optional[Tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_pool, src_txt_ids = self.prepare_embed(src_prompt, src_prompt_emb)
            tgt_emb, tgt_pool, tgt_txt_ids = self.prepare_embed(tgt_prompt, tgt_prompt_emb)

        with torch.no_grad():
            z_src, img_ids, h_lat, w_lat = self.prepare_latents(src_img)

        z_tgt = z_src.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([z_tgt], lr=lr_max)

        sigmas = np.linspace(1.0, 1 / steps, steps)
        mu = calculate_shift(
            z_src.shape[1],
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift
        )
        timesteps, _ = retrieve_timesteps(
            self.scheduler, steps, self.device, timesteps=None, sigmas=sigmas, mu=mu
        )

        device = self.device if not self.offload else "cuda"
        tar_guidance = torch.tensor([tar_cfg_scale], device=device).expand(z_src.shape[0])
        src_guidance = torch.tensor([src_cfg_scale], device=device).expand(z_src.shape[0])

        alpha_T_steps = (timesteps[steps - 2].item() - timesteps[steps - 1].item()) / 1.6
        alpha_max_dynamic = alpha_T_steps / 1.6 if alpha_T_steps > 0 else lr_max
        beta_tail = alpha_T_steps / 4 if alpha_T_steps > 0 else lr_max / 4

        pbar = tqdm(enumerate(timesteps), total=steps, desc="FLUX DVRF")
        for i, t in pbar:
            t_curr = t
            current_lr = 2.2 * lr_hump_tail_beta(i + 1, steps + 28, alpha_max_dynamic, beta_tail, a=10, b=8)
            optimizer.param_groups[0]["lr"] = current_lr

            eta_i = eta * i / steps
            eps = torch.randn_like(z_src)
            qt = (1 - t_curr) * z_src + t_curr * eps

            z_tgt_detached = z_tgt.detach()
            shift_term = eta_i * t_curr * (z_tgt_detached - z_src)
            pt = (1 - t_curr) * z_tgt_detached + t_curr * eps + shift_term

            t_tensor = t_curr.view(1)
            with torch.inference_mode():
                v_tar = self.predict_vector(pt, t_tensor, tgt_emb, tgt_pool, tgt_txt_ids, img_ids, guidance=tar_guidance)
                v_src = self.predict_vector(qt, t_tensor, src_emb, src_pool, src_txt_ids, img_ids, guidance=src_guidance)

            delta_v = v_tar - v_src
            geometric_term = (1 - eta_i) * (z_tgt_detached - z_src)
            grad = delta_v + geometric_term

            loss = (z_tgt * grad.detach()).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"lr": f"{current_lr:.5f}", "eta": f"{eta_i:.2f}"})

        with torch.no_grad():
            img = self.decode(z_tgt.detach(), h_lat, w_lat)
        return img
