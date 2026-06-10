from typing import List, Tuple, Optional
import torch
from diffusers import StableDiffusion3Pipeline

__SAMPLER__ = {}


def register_sampler(name: str) -> callable:
    def wrapper(cls):
        if __SAMPLER__.get(name, None) is not None:
            raise ValueError(f"Sampler {name} already registered")
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sd3_sampler(name: str, **kwargs) -> object:
    if name not in __SAMPLER__:
        raise ValueError(f"Sampler {name} does not exist")
    return __SAMPLER__[name](**kwargs)


class StableDiffusion3Base:
    def __init__(self,
                 model_key: str = "stabilityai/stable-diffusion-3-medium-diffusers",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 offload: bool = False) -> None:
        self.device = device
        self.dtype = dtype
        self.offload = offload

        print(f"Loading SD3 Base from {model_key} (Offload: {offload})...")
        pipe = StableDiffusion3Pipeline.from_pretrained(model_key, torch_dtype=self.dtype)

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        elif hasattr(pipe.vae, "enable_slicing"):
            try:
                pipe.vae.enable_slicing()
            except:
                pass

        if self.offload:
            pipe.enable_model_cpu_offload()

        self.scheduler = pipe.scheduler

        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3

        self.text_enc_1 = pipe.text_encoder
        self.text_enc_1.eval()
        self.text_enc_1.requires_grad_(False)
        self.text_enc_2 = pipe.text_encoder_2
        self.text_enc_2.eval()
        self.text_enc_2.requires_grad_(False)
        self.text_enc_3 = pipe.text_encoder_3
        self.text_enc_3.eval()
        self.text_enc_3.requires_grad_(False)

        self.vae = pipe.vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        if not self.offload:
            self.text_enc_1.to(device)
            self.text_enc_2.to(device)
            self.text_enc_3.to(device)
            self.vae.to(device)
            self.transformer.to(device)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        del pipe

    def encode_prompt(self, prompt: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device if not self.offload else "cuda"

        text_t5_ids = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        ).input_ids
        text_t5_emb = self.text_enc_3(text_t5_ids.to(device))[0].to(dtype=self.dtype, device=device)

        text_clip1_ids = self.tokenizer_1(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        text_clip1_emb = self.text_enc_1(text_clip1_ids.to(device), output_hidden_states=True)
        pool_clip1_emb = text_clip1_emb[0].to(dtype=self.dtype, device=device)
        text_clip1_emb = text_clip1_emb.hidden_states[-2].to(dtype=self.dtype, device=device)

        text_clip2_ids = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).input_ids
        text_clip2_emb = self.text_enc_2(text_clip2_ids.to(device), output_hidden_states=True)
        pool_clip2_emb = text_clip2_emb[0].to(dtype=self.dtype, device=device)
        text_clip2_emb = text_clip2_emb.hidden_states[-2].to(dtype=self.dtype, device=device)

        clip_prompt_emb = torch.cat([text_clip1_emb, text_clip2_emb], dim=-1)
        clip_prompt_emb = torch.nn.functional.pad(clip_prompt_emb, (0, text_t5_emb.shape[-1] - clip_prompt_emb.shape[-1]))

        prompt_emb = torch.cat([clip_prompt_emb, text_t5_emb], dim=-2)
        pooled_prompt_emb = torch.cat([pool_clip1_emb, pool_clip2_emb], dim=-1)

        return prompt_emb, pooled_prompt_emb

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        device = self.device if not self.offload else "cuda"
        image = image.to(device=device, dtype=self.dtype)

        z = self.vae.encode(image).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self,
                       z: torch.Tensor,
                       t: torch.Tensor,
                       prompt_emb: torch.Tensor,
                       pooled_emb: torch.Tensor) -> torch.Tensor:
        if len(t.shape) == 0:
            t = t.expand(z.shape[0])
        t = t.to(dtype=self.dtype)

        if self.offload:
            z = z.to("cuda")
            t = t.to("cuda")
            prompt_emb = prompt_emb.to("cuda")
            pooled_emb = pooled_emb.to("cuda")

        v = self.transformer(
            hidden_states=z,
            timestep=t,
            pooled_projections=pooled_emb,
            encoder_hidden_states=prompt_emb,
            return_dict=False
        )[0]
        return v

    def prepare_embed(self,
                      prompt: str,
                      embs: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        if embs is None:
            return self.encode_prompt([prompt])

        device = self.device if not self.offload else "cuda"
        return embs[0].to(device), embs[1].to(device)