from typing import Dict, Any, Tuple, Optional

class Config:
    CSV_PATH = "data/flowedit_dataset.csv"
    YAML_PATH = "data/flowedit_prompt.yaml"
    IMAGE_DIR = "data/flowedit_images"

    OUTPUT_SRC_DIR = "data/outputs/source"
    OUTPUT_TAR_DIR = "data/outputs/target"

    MODEL = {
        "SD3": {
            "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
            "T_steps": 50,
            "n_avg": 1,
            "src_guidance": 3.5,
            "tar_guidance": 13.5,
            "n_min": 0,
            "n_max": 33,
            "seed": 42
        },
        "FLUX": {
            "model_id": "black-forest-labs/FLUX.1-dev",
            "T_steps": 28,
            "n_avg": 1,
            "src_guidance": 1.5,
            "tar_guidance": 5.5,
            "n_min": 0,
            "n_max": 24,
            "seed": 42
        },
        "INSTAFLOW": {
            "model_id": "XCLiu/2_rectified_flow_from_sd_1_5",
            "T_steps": 25,
            "n_avg": 1,
            "src_guidance": 1.5, 
            "tar_guidance": 16.5, 
            "n_min": 0,          
            "n_max": 21,         
            "seed": 42
        }
    }

    @classmethod
    def get_csv_path(self) -> str:
        return self.CSV_PATH

    @classmethod
    def set_csv_path(self, path: str) -> None:
        self.CSV_PATH = path

    @classmethod
    def get_yaml_path(self) -> str:
        return self.YAML_PATH

    @classmethod
    def set_yaml_path(self, path: str) -> None:
        self.YAML_PATH = path
    
    @classmethod
    def get_output_dirs(self) -> Tuple[str, str]:
        return self.OUTPUT_SRC_DIR, self.OUTPUT_TAR_DIR
    
    @classmethod
    def set_output_dirs(self, path1: str, path2: str) -> None:
        self.OUTPUT_SRC_DIR = path1
        self.OUTPUT_TAR_DIR = path2

    @classmethod
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Lấy config dict của model. Trả về None nếu không tìm thấy
        """
        return self.MODEL.get(model_name, None)

    @classmethod
    def get_param(self, model_name: str, param_key: str) -> Any:
        """
        Lấy một tham số cụ thể
        """
        if model_name in self.MODEL:
            return self.MODEL[model_name].get(param_key, None)
        return None

    @classmethod
    def set_param(self, model_name: str, param_key: str, value: Any) -> None:
        """
        Cập nhật tham số
        """
        if model_name in self.MODEL:
            self.MODEL[model_name][param_key] = value
            print(f"Updated {model_name} [{param_key}] = {value}")
        else:
            print(f"Model {model_name} not found")

    @classmethod
    def add_new_model(self, model_name: str, config_dict: Dict[str, Any]) -> None:
        """
        Thêm một cấu hình model mới vào dictionary
        """
        self.CONFIG[model_name] = config_dict