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
            # FireFlow (arXiv:2412.07517): modified midpoint solver + V-feature injection.
            # NFE here is steps per phase; total NFE ≈ 2*(NFE+1).
            # cfg_scale during denoising; inversion always uses guidance=1 (no CFG).
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
            # guidance_scale controls the L2 structure-preservation gradient strength
            # (s' in Theorem 4.3 of arXiv:2412.00100). Higher = more source fidelity.
            # cfg_scale drives the edit via the target text prompt.
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
        "uniedit": {
            # UniEdit-Flow (arXiv:2504.13109): N=total steps, alpha=delay rate,
            # omega=velocity-difference guidance strength.
            # NFE = 3*alpha*N + 1  (N=15, alpha=0.6 → 28 NFE).
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
    }

    @classmethod
    def get_csv_path(self) -> str:
        return self.CSV_PATH

    @classmethod
    def set_csv_path(self, path: str) -> None:
        self.CSV_PATH = path

    @classmethod
    def get_image_dir(self) -> str:
        return self.IMAGE_DIR

    @classmethod
    def set_image_dir(self, path: str) -> None:
        self.IMAGE_DIR = path

    @classmethod
    def get_output_dir(self) -> str:
        return self.OUTPUT_DIR

    @classmethod
    def set_output_dir(self, path: str) -> None:
        self.OUTPUT_DIR = path

    @classmethod
    def get_params(self, 
                   algorithm: str, 
                   model_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full dictionary of parameters for a specific algorithm and model
        """
        algo_key = algorithm.lower()
        model_key = model_name.lower()
        
        algo_params = self.MODEL_PARAMS.get(algo_key)
        if algo_params:
            params = algo_params.get(model_key)
            return params.copy() if params else None
        return None

    @classmethod
    def set_param(self, 
                  algorithm: str, 
                  model_name: str, 
                  param_key: str, 
                  value: Any) -> bool:
        """
        Updates a specific parameter for a model configuration
        
        Parameters:
            algorithm: "flowedit" or "flowalign"
            model_name: "sd3", "flux", or "instaflow"
            param_key: The specific parameter to change (e.g., "NFE", "cfg_scale")
            value: The new value
            
        Returns:
            bool: True if updated successfully, False if model/algo not found
        """
        algo_key = algorithm.lower()
        model_key = model_name.lower()

        if algo_key in self.MODEL_PARAMS and model_key in self.MODEL_PARAMS[algo_key]:
            self.MODEL_PARAMS[algo_key][model_key][param_key] = value
            return True
        return False

    @classmethod
    def get_param_value(self, 
                        algorithm: str, 
                        model_name: str, 
                        param_key: str) -> Any:
        """
        Gets a single value of a specific parameter
        """
        algo_key = algorithm.lower()
        model_key = model_name.lower()
        
        if algo_key in self.MODEL_PARAMS and model_key in self.MODEL_PARAMS[algo_key]:
            return self.MODEL_PARAMS[algo_key][model_key].get(param_key)
        return None