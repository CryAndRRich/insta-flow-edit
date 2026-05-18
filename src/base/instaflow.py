from typing import Tuple, Optional
import torch
from diffusers import StableDiffusionPipeline

__SAMPLER__ = {}


def register_sampler(name: str) -> callable:
    def wrapper(cls):
        if __SAMPLER__.get(name, None) is not None:
            raise ValueError(f"Sampler {name} already registered.")
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_instaflow_sampler(name: str, **kwargs) -> object:
    if name not in __SAMPLER__:
        raise ValueError(f"Sampler {name} does not exist.")
    return __SAMPLER__[name](**kwargs)


class InstaFlowBase:
    def __init__(self,
                 model_key: str = "XCLiu/2_rectified_flow_from_sd_1_5",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 offload: bool = False) -> None:
        self.device = device
        self.dtype = dtype
        self.offload = offload

        print(f"Loading InstaFlow Base from {model_key} (Offload: {offload})...")

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype)
        pipe.safety_checker = None

        if self.offload:
            pipe.enable_model_cpu_offload()

        self.scheduler = pipe.scheduler

        self.vae = pipe.vae
        self.vae.eval()
        self.vae.requires_grad_(False)

        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        self.unet = pipe.unet
        self.unet.eval()
        self.unet.requires_grad_(False)

        if not self.offload:
            self.vae.to(device)
            self.text_encoder.to(device)
            self.unet.to(device)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        del pipe

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        device = self.device if not self.offload else "cuda"

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        prompt_embeds = self.text_encoder(text_input_ids)[0]
        return prompt_embeds.to(dtype=self.dtype, device=device)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        device = self.device if not self.offload else "cuda"
        image = image.to(device=device, dtype=self.dtype)

        z = self.vae.encode(image).latent_dist.sample()
        z = z * self.vae.config.scaling_factor
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.vae.config.scaling_factor
        return self.vae.decode(z, return_dict=False)[0]

    def predict_vector(self,
                       z: torch.Tensor,
                       t: torch.Tensor,
                       prompt_embeds: torch.Tensor) -> torch.Tensor:
        timestep = t * self.scheduler.config.num_train_timesteps
        timestep = timestep.to(z.device).to(z.dtype)
        if len(timestep.shape) == 0:
            timestep = timestep.expand(z.shape[0])

        if self.offload:
            z = z.to("cuda")
            timestep = timestep.to("cuda")
            prompt_embeds = prompt_embeds.to("cuda")

        v = self.unet(
            z,
            timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        return v

    def prepare_embed(self,
                      prompt: str,
                      neg_prompt: str = "",
                      embs: Optional[torch.Tensor] = None,
                      neg_embs: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if embs is None:
            embs = self.encode_prompt(prompt)

        if neg_embs is None:
            neg_embs = self.encode_prompt(neg_prompt)

        device = self.device if not self.offload else "cuda"
        return embs.to(device), neg_embs.to(device)
