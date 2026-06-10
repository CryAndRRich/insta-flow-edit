from typing import Tuple, Optional
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel, BitsAndBytesConfig

__SAMPLER__ = {}


def register_sampler(name: str) -> callable:
    def wrapper(cls):
        if __SAMPLER__.get(name, None) is not None:
            raise ValueError(f"Sampler {name} already registered.")
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_flux_sampler(name: str, **kwargs) -> object:
    if name not in __SAMPLER__:
        raise ValueError(f"Sampler {name} does not exist.")
    return __SAMPLER__[name](**kwargs)


class FluxBase:
    def __init__(self,
                 model_key: str = "black-forest-labs/FLUX.1-dev",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 offload: bool = False,
                 quantize_4bit: bool = True) -> None:
        self.device = device
        self.dtype = dtype
        self.offload = offload

        print(f"Loading FLUX Base from {model_key} (Offload: {offload}, 4-bit: {quantize_4bit})...")

        if quantize_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype
            )

            transformer_device_map = "auto" if not offload else None

            transformer = FluxTransformer2DModel.from_pretrained(
                model_key,
                subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=dtype,
                device_map=transformer_device_map
            )

            text_encoder_2 = T5EncoderModel.from_pretrained(
                model_key,
                subfolder="text_encoder_2",
                quantization_config=quant_config,
                torch_dtype=dtype,
                device_map=transformer_device_map
            )

            pipe = FluxPipeline.from_pretrained(
                model_key,
                transformer=transformer,
                text_encoder_2=text_encoder_2,
                torch_dtype=dtype
            )
        else:
            pipe = FluxPipeline.from_pretrained(model_key, torch_dtype=dtype)

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()

        if self.offload:
            pipe.enable_model_cpu_offload()
        else:
            if not quantize_4bit:
                pipe.to(device)
            else:
                pipe.text_encoder.to(device)
                pipe.vae.to(device)

        self.scheduler = pipe.scheduler

        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2

        self.text_encoder = pipe.text_encoder
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_2.eval()
        self.text_encoder_2.requires_grad_(False)

        self.vae = pipe.vae
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.transformer = pipe.transformer
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.vae_scale_factor = pipe.vae_scale_factor

        del pipe

    def encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.device if not self.offload else "cuda"

        text_clip_ids = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        pooled_prompt_embeds = self.text_encoder(text_clip_ids.to(device), output_hidden_states=False).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=self.dtype, device=device)

        text_t5_ids = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        prompt_embeds = self.text_encoder_2(text_t5_ids.to(device), output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.dtype, device=device)

        txt_ids = torch.zeros(prompt_embeds.shape[0], prompt_embeds.shape[1], 3).to(device=device, dtype=self.dtype)

        return prompt_embeds, pooled_prompt_embeds, txt_ids

    def prepare_latents(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        device = self.device if not self.offload else "cuda"
        image = image.to(device=device, dtype=self.dtype)

        z = self.vae.encode(image).latent_dist.sample()
        z = (z - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        h_lat, w_lat = z.shape[2], z.shape[3]

        batch_size, num_channels, height, width = z.shape
        z_packed = self._pack_latents(z, batch_size, num_channels, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, self.dtype)

        return z_packed, latent_image_ids, h_lat, w_lat

    def _pack_latents(self,
                      latents: torch.Tensor,
                      batch_size: int,
                      num_channels_latents: int,
                      height: int,
                      width: int) -> torch.Tensor:
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    def _unpack_latents(self,
                        latents: torch.Tensor,
                        height: int,
                        width: int) -> torch.Tensor:
        batch_size = latents.shape[0]
        num_channels_latents = (latents.shape[2] // 4)

        latents = latents.view(batch_size, height // 2, width // 2, num_channels_latents, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, num_channels_latents, height, width)
        return latents

    def _prepare_latent_image_ids(self,
                                  batch_size: int,
                                  height: int,
                                  width: int,
                                  device: str,
                                  dtype: torch.dtype) -> torch.Tensor:
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1)
        latent_image_ids = latent_image_ids.reshape(batch_size, (height // 2) * (width // 2), 3)
        return latent_image_ids.to(device=device, dtype=dtype)

    def decode(self,
               z_packed: torch.Tensor,
               h_lat: int,
               w_lat: int) -> torch.Tensor:
        z = self._unpack_latents(z_packed, h_lat, w_lat)
        z = (z / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self,
                       z: torch.Tensor,
                       t: torch.Tensor,
                       prompt_emb: torch.Tensor,
                       pooled_emb: torch.Tensor,
                       txt_ids: torch.Tensor,
                       img_ids: torch.Tensor,
                       guidance: Optional[torch.Tensor] = None) -> torch.Tensor:
        if len(t.shape) == 0:
            t = t.expand(z.shape[0])
        t = t.to(dtype=self.dtype)

        if self.offload:
            z = z.to("cuda")
            t = t.to("cuda")
            prompt_emb = prompt_emb.to("cuda")
            pooled_emb = pooled_emb.to("cuda")
            txt_ids = txt_ids.to("cuda")
            img_ids = img_ids.to("cuda")
            if guidance is not None:
                guidance = guidance.to("cuda")

        v = self.transformer(
            hidden_states=z,
            timestep=t,
            guidance=guidance,
            encoder_hidden_states=prompt_emb,
            pooled_projections=pooled_emb,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False
        )[0]

        return v

    def prepare_embed(self,
                      prompt: str,
                      embs: Optional[Tuple]) -> Tuple:
        if embs is None:
            return self.encode_prompt(prompt)
        device = self.device if not self.offload else "cuda"
        return tuple(e.to(device) for e in embs)