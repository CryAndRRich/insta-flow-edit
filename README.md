<div align="center">

# InstaFlowEdit
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![FLUX.1-dev](https://img.shields.io/badge/🤗%20FLUX.1--dev-black--forest--labs-FFD21E)](https://huggingface.co/black-forest-labs/FLUX.1-dev)
[![SD3](https://img.shields.io/badge/🤗%20SD3-stabilityai-FFD21E)](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
[![InstaFlow](https://img.shields.io/badge/🤗%20InstaFlow-XCLiu-FFD21E)](https://huggingface.co/XCLiu/2_rectified_flow_from_sd_1_5)

</div>

## Overview

**insta-flow-edit** provides clean, plug-and-play implementations of state-of-the-art **training-free text-based image editing** algorithms, all unified under a single API and running on top of **flow-matching / rectified-flow generative models**

Flow-matching models parameterise a straight-line ODE between data $x_0$ and noise $\varepsilon$:

$$x_t = (1-t)\,x_0 + t\,\varepsilon, \qquad v_\theta(x_t,\,t) \approx \varepsilon - x_0$$

This linearity makes them uniquely suited for lightweight, inversion-free editing: the source latent trajectory can be offset, injected into, or gradient-guided toward a target semantics without any fine-tuning

## Backbones Models

| Model | Architecture | Native Resolution | Link |
|----------|-------------|:-----------------:|:-----------:|
| FLUX.1-dev | Diffusion Transformer (DiT) | 1024 × 1024 | [[GitHub](https://github.com/black-forest-labs/flux)] [[HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev)] |
| Stable Diffusion 3 | Multimodal DiT (MMDiT) | 1024 × 1024 | [[GitHub](https://github.com/Stability-AI/sd3-ref)] [[HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)] |
| InstaFlow | UNet (2-rectified flow) | 512 × 512 | [[GitHub](https://github.com/gnobitab/InstaFlow)] [[arXiv](https://arxiv.org/abs/2309.06380)]  [[HuggingFace](https://huggingface.co/XCLiu/2_rectified_flow_from_sd_1_5)] |

---

## Editing Methods

| Method | FLUX | SD3 | InstaFlow | Link |
|--------|:----:|:---:|:---------:|:----:|
| **FlowEdit** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2412.08629)] [[GitHub](https://github.com/fallenshock/FlowEdit)] [[ProjectPage](https://matankleiner.github.io/flowedit/)] |
| **FireFlow** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2412.07517)] [[GitHub](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing)] |
| **FlowChef** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2412.00100)] [[GitHub](https://github.com/FlowChef/FlowChef)] [[ProjectPage](https://flowchef.github.io/)] |
| **UniEdit-Flow** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2504.13109)] [[GitHub](https://github.com/DSL-Lab/UniEdit-Flow)] [[ProjectPage](https://uniedit-flow.github.io/)] |
| **FlowAlign** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2505.23145)] [[GitHub](https://github.com/FlowAlign/FlowAlign)] |
| **TweezeEdit** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2508.10498)] [[GitHub](https://github.com/hdsfade/TweezeEdit)] |
| **DVRF** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2509.05342)] [[GitHub](https://github.com/Harvard-AI-and-Robotics-Lab/DeltaRectifiedFlowSampling)] |
| **CVC** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2512.24015)] |
| **ChordEdit** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2602.19083)] [[GitHub](https://github.com/ChordEdit/ChordEdit)] [[ProjectPage](https://chordedit.github.io/)] |
| **VeloEdit** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2603.13388)] [[GitHub](https://github.com/xmulzq/VeloEdit)] |
| **FlowSlider** | [x] | [x] | [x] | [[arXiv](https://arxiv.org/abs/2604.02088)] [[HuggingFace](https://huggingface.co/spaces/dominoer/FlowSlider)] |

## Quick Start

### Installation

```bash
pip install torch torchvision diffusers transformers accelerate \
            sentencepiece protobuf tqdm pillow
```

### Data Format

Place source images in `data/images/` and create `data/dataset.csv`:

```csv
name,source_prompt,target_prompts
cat,"a photo of a cat sitting on a sofa","a photo of a dog sitting on a sofa"
```

### Usage

```python
import torch
from src import get_sampler
from utils.load_data import load_data

# Load image + prompts (use resize_size=512 for InstaFlow)
image, src_prompt, tgt_prompt = load_data(
    image_dir="data/images",
    csv_path="data/dataset.csv",
    image_name="cat.jpg",
    resize_size=1024,
)

# Pick backbone ("flux" | "sd3" | "instaflow") and method
model = get_sampler("flux", "flowedit", model_key="black-forest-labs/FLUX.1-dev")

# Edit
result = model.sample(image, src_prompt, tgt_prompt, NFE=28, tar_cfg_scale=5.5, src_cfg_scale=1.5)
```

Switch backbone or method with a single line:

```python
model = get_sampler("sd3", "fireflow", model_key="stabilityai/stable-diffusion-3-medium-diffusers")
model = get_sampler("instaflow", "flowalign", model_key="XCLiu/2_rectified_flow_from_sd_1_5")
model = get_sampler("flux", "tweezeedit", model_key="black-forest-labs/FLUX.1-dev")
model = get_sampler("sd3", "chordedit", model_key="stabilityai/stable-diffusion-3-medium-diffusers")
```

Pre-tuned hyperparameters for every (method, backbone) pair are available in [`configs/config.py`](configs/config.py).

---

## Acknowledgements

This project adapts and unifies implementations from the following works:

[FlowEdit](https://github.com/fallenshock/FlowEdit) - [FireFlow](https://github.com/HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing) - [FlowChef](https://github.com/FlowChef/FlowChef) - [UniEdit-Flow](https://github.com/DSL-Lab/UniEdit-Flow) - [FlowAlign](https://github.com/FlowAlign/FlowAlign) - [TweezeEdit](https://github.com/hdsfade/TweezeEdit) - [DVRF](https://github.com/Harvard-AI-and-Robotics-Lab/DeltaRectifiedFlowSampling) - [CVC](https://arxiv.org/abs/2512.24015) - [ChordEdit](https://github.com/ChordEdit/ChordEdit) - [VeloEdit](https://github.com/xmulzq/VeloEdit) - [FlowSlider](https://huggingface.co/spaces/dominoer/FlowSlider)
