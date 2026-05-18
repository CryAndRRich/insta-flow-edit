from .base.flux import get_flux_sampler
from .base.sd3 import get_sd3_sampler
from .base.instaflow import get_instaflow_sampler

# Import all method packages to trigger @register_sampler decorators
from . import flowedit, flowalign, dvrf, fireflow, uniedit, flowchef


def get_sampler(sampler_name: str,
                edit_name: str,
                **kwargs) -> object:
    samplers = {
        "flux": get_flux_sampler,
        "sd3": get_sd3_sampler,
        "instaflow": get_instaflow_sampler,
    }
    if sampler_name in samplers:
        return samplers[sampler_name](name=edit_name, **kwargs)
    else:
        raise ValueError(f"Sampler '{sampler_name}' not recognized")


__all__ = ["get_sampler"]
