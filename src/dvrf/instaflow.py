from typing import Optional
import torch
from tqdm.auto import tqdm
from ..base.instaflow import InstaFlowBase, register_sampler
from ..base.helper import lr_hump_tail_beta


@register_sampler(name="dvrf")
class InstaDeltaVelFlow(InstaFlowBase):
    def sample(self,
               src_img: torch.Tensor,
               src_prompt: str,
               tgt_prompt: str,
               neg_prompt: str = "",
               steps: int = 50,
               eta: float = 1.0,
               lr_max: float = 0.04,
               tar_cfg_scale: float = 16.5,
               src_cfg_scale: float = 6.0,
               src_prompt_emb: Optional[torch.Tensor] = None,
               tgt_prompt_emb: Optional[torch.Tensor] = None,
               neg_prompt_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            src_emb, src_neg = self.prepare_embed(src_prompt, neg_prompt, src_prompt_emb, neg_prompt_emb)
            tgt_emb, tgt_neg = self.prepare_embed(tgt_prompt, neg_prompt, tgt_prompt_emb, neg_prompt_emb)
            combined_embeds = torch.cat([src_neg, src_emb, tgt_neg, tgt_emb], dim=0)

        with torch.no_grad():
            z_src = self.encode(src_img)

        z_tgt = z_src.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([z_tgt], lr=lr_max)

        timesteps = torch.linspace(1.0, 0.0, steps + 1)[:-1].to(self.device)

        alpha_T_steps = (1.0 / steps) / 1.6
        alpha_max_dynamic = alpha_T_steps / 1.6 if alpha_T_steps > 0 else lr_max
        beta_tail = alpha_T_steps / 4 if alpha_T_steps > 0 else lr_max / 4

        pbar = tqdm(enumerate(timesteps), total=steps, desc="InstaFlow DVRF Optimization")
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

            with torch.inference_mode():
                latent_input = torch.cat([qt, qt, pt, pt])
                v_all = self.predict_vector(latent_input, t_curr, combined_embeds)
                v_src_neg, v_src_cond, v_tar_neg, v_tar_cond = v_all.chunk(4)
                v_src = v_src_neg + src_cfg_scale * (v_src_cond - v_src_neg)
                v_tar = v_tar_neg + tar_cfg_scale * (v_tar_cond - v_tar_neg)

            delta_v = v_tar - v_src
            geometric_term = (1 - eta_i) * (z_tgt_detached - z_src)
            grad = delta_v + geometric_term

            loss = (z_tgt * grad.detach()).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"lr": f"{current_lr:.5f}", "eta": f"{eta_i:.2f}"})

        with torch.no_grad():
            img = self.decode(z_tgt.detach())
        return img