from typing import Dict, Any, Optional


class Config:
    """Configuration class to manage file paths and model parameters"""

    CSV_PATH: str = "data/dataset.csv"
    IMAGE_DIR: str = "data/images"
    OUTPUT_DIR: str = "data/output_images"

    MODEL_PARAMS: Dict[str, Dict[str, Dict[str, Any]]] = {
        "flowedit": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "NFE": 50,
                "n_start": 17,
                "tar_cfg_scale": 13.5,
                "src_cfg_scale": 3.5,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "NFE": 28,
                "n_start": 4,
                "tar_cfg_scale": 5.5,
                "src_cfg_scale": 1.5,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "NFE": 25,
                "n_start": 4,
                "tar_cfg_scale": 16.5,
                "src_cfg_scale": 1.5,
            }
        },
        "flowalign": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "NFE": 50,
                "n_start": 17,
                "cfg_scale": 7.0,
                "zeta": 0.01,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "NFE": 28,
                "n_start": 4,
                "cfg_scale": 3.5,
                "zeta": 0.01,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "NFE": 25,
                "n_start": 4,
                "cfg_scale": 16.5,
                "zeta": 0.01,
            }
        },
        "dvrf": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "steps": 50,
                "eta": 1.0,
                "lr_max": 0.04,
                "tar_cfg_scale": 16.5,
                "src_cfg_scale": 6.0,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "steps": 50,
                "eta": 1.0,
                "lr_max": 0.04,
                "tar_cfg_scale": 5.5,
                "src_cfg_scale": 1.5,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "steps": 50,
                "eta": 1.0,
                "lr_max": 0.04,
                "tar_cfg_scale": 16.5,
                "src_cfg_scale": 1.5,
            }
        },
        "fireflow": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "NFE": 28,
                "cfg_scale": 13.5,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "NFE": 8,
                "cfg_scale": 5.0,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "NFE": 25,
                "cfg_scale": 7.5,
            },
        },
        "flowchef": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "NFE": 30,
                "guidance_scale": 10.0,
                "cfg_scale": 7.5,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "NFE": 28,
                "guidance_scale": 7.5,
                "cfg_scale": 5.5,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "NFE": 25,
                "guidance_scale": 15.0,
                "cfg_scale": 10.0,
            }
        },
        "cvc": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "NFE": 50,
                "alpha": 1.0,
                "beta": 7.0,
                "eta": 0.2,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "NFE": 28,
                "alpha": 1.0,
                "beta": 3.5,
                "eta": 0.2,
                "cfg_scale": 3.5,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "NFE": 25,
                "alpha": 1.0,
                "beta": 7.0,
                "eta": 0.2,
            },
        },
        "uniedit": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "N": 15,
                "alpha": 0.6,
                "omega": 5.0,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "N": 15,
                "alpha": 0.6,
                "omega": 5.0,
                "cfg_scale": 3.5,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "N": 15,
                "alpha": 0.6,
                "omega": 5.0,
            },
        },
        "tweezeedit": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "NFE": 50,
                "gamma": 1.0,
                "tgt_cfg_scale": 13.5,
                "src_cfg_scale": 3.5,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "NFE": 28,
                "gamma": 1.0,
                "tgt_cfg_scale": 5.5,
                "src_cfg_scale": 1.5,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "NFE": 25,
                "gamma": 1.0,
                "tgt_cfg_scale": 16.5,
                "src_cfg_scale": 1.5,
            },
        },
        "chordedit": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "sigma_t": 0.90,
                "delta": 0.15,
                "lambda_": 1.0,
                "sigma_c": 0.30,
                "cfg_scale": 7.5,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "sigma_t": 0.90,
                "delta": 0.15,
                "lambda_": 1.0,
                "sigma_c": 0.30,
                "cfg_scale": 3.5,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "sigma_t": 0.90,
                "delta": 0.15,
                "lambda_": 1.0,
                "sigma_c": 0.30,
                "cfg_scale": 7.5,
            },
        },
        "veloedit": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "NFE": 50,
                "N": 1,
                "tau": 0.4,
                "alpha": 0.8,
                "cfg_scale": 13.5,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "NFE": 28,
                "N": 1,
                "tau": 0.4,
                "alpha": 0.8,
                "cfg_scale": 5.5,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "NFE": 25,
                "N": 1,
                "tau": 0.4,
                "alpha": 0.8,
                "cfg_scale": 7.5,
            },
        },
        "flowslider": {
            "sd3": {
                "model_key": "stabilityai/stable-diffusion-3-medium-diffusers",
                "NFE": 50,
                "n_start": 0,
                "strength": 1.0,
                "cfg_scale": 3.5,
            },
            "flux": {
                "model_key": "black-forest-labs/FLUX.1-dev",
                "NFE": 28,
                "n_start": 0,
                "strength": 1.0,
                "cfg_scale": 3.5,
            },
            "instaflow": {
                "model_key": "XCLiu/2_rectified_flow_from_sd_1_5",
                "NFE": 25,
                "n_start": 0,
                "strength": 1.0,
                "cfg_scale": 7.5,
            },
        },
    }

    @classmethod
    def get_csv_path(cls) -> str:
        return cls.CSV_PATH

    @classmethod
    def set_csv_path(cls, path: str) -> None:
        cls.CSV_PATH = path

    @classmethod
    def get_image_dir(cls) -> str:
        return cls.IMAGE_DIR

    @classmethod
    def set_image_dir(cls, path: str) -> None:
        cls.IMAGE_DIR = path

    @classmethod
    def get_output_dir(cls) -> str:
        return cls.OUTPUT_DIR

    @classmethod
    def set_output_dir(cls, path: str) -> None:
        cls.OUTPUT_DIR = path

    @classmethod
    def get_params(cls,
                   algorithm: str,
                   model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full dictionary of parameters for a specific algorithm and model
        """
        algo_key = algorithm.lower()
        model_key = model_name.lower()

        algo_params = cls.MODEL_PARAMS.get(algo_key)
        if algo_params:
            params = algo_params.get(model_key)
            return params.copy() if params else None
        return None

    @classmethod
    def set_param(cls,
                  algorithm: str,
                  model_name: str,
                  param_key: str,
                  value: Any) -> bool:
        """
        Updates a specific parameter for a model configuration

        Parameters:
            algorithm: algorithm name (e.g. "flowedit", "flowalign", "fireflow", ...)
            model_name: "sd3", "flux", or "instaflow"
            param_key: The specific parameter to change (e.g., "NFE", "cfg_scale")
            value: The new value

        Returns:
            bool: True if updated successfully, False if model/algo not found
        """
        algo_key = algorithm.lower()
        model_key = model_name.lower()

        if algo_key in cls.MODEL_PARAMS and model_key in cls.MODEL_PARAMS[algo_key]:
            cls.MODEL_PARAMS[algo_key][model_key][param_key] = value
            return True
        return False

    @classmethod
    def get_param_value(cls,
                        algorithm: str,
                        model_name: str,
                        param_key: str) -> Any:
        """
        Gets a single value of a specific parameter
        """
        algo_key = algorithm.lower()
        model_key = model_name.lower()

        if algo_key in cls.MODEL_PARAMS and model_key in cls.MODEL_PARAMS[algo_key]:
            return cls.MODEL_PARAMS[algo_key][model_key].get(param_key)
        return None