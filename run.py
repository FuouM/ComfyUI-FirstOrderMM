from pathlib import Path

import torch

from .constants import default_model_name, supported_model_names
from .utils import (
    get_config_path,
    get_model_file_name,
    inference,
    inference_best_frame,
    load_checkpoint,
    out_video,
    reshape_image,
)

base_dir = Path(__file__).resolve().parent


class FOMM_Runner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "driving_video_input": ("IMAGE",),
                "model_name": (supported_model_names, {"default": default_model_name}),
                "frame_rate": ("FLOAT", {"default": 30.0}),
                "relative_movement": (
                    "BOOLEAN",
                    {"default": True},
                ),  # Relative keypoint displacement (Inherit object proporions from the video)
                "relative_jacobian": (
                    "BOOLEAN",
                    {"default": True},
                ),  # No idea
                "adapt_movement_scale": (
                    "BOOLEAN",
                    {"default": True},
                ),  # Adapt movement scale
                "find_best_frame": (
                    "BOOLEAN",
                    {"default": False},
                ),  # Generate from the frame that is the most alligned with source
            },
            "optional": {"audio": ("VHS_AUDIO",)},
        }

    RETURN_TYPES = (
        "IMAGE",
        "VHS_AUDIO",
        "FLOAT",
    )
    RETURN_NAMES = (
        "images",
        "audio",
        "frame_rate",
    )
    FUNCTION = "todo"
    CATEGORY = "FirstOrderMM"

    def todo(
        self,
        source_image: torch.Tensor,
        driving_video_input: torch.Tensor,
        model_name: str,
        frame_rate: float,
        relative_movement: bool,
        relative_jacobian: bool,
        adapt_movement_scale: bool,
        find_best_frame: bool,
        audio=None,
    ):
        print(f"{type(source_image)=}")
        print(f"{type(driving_video_input)=}")
        print(f"{source_image.shape=}")
        print(f"{driving_video_input.shape=}")
        print(f"{type(audio)=}")
        print(base_dir)

        config_path = f"{base_dir}/{get_config_path(model_name)}"
        checkpoint_path = f"{base_dir}/{get_model_file_name(model_name)}"

        generator, kp_detector = load_checkpoint(config_path, checkpoint_path)
        source_image = reshape_image(source_image, (256, 256))
        driving_video = reshape_image(driving_video_input, (256, 256)).unsqueeze(0)
        driving_video = driving_video.permute(0, 2, 1, 3, 4)

        print("After reshaping")
        print(f"{source_image.shape=}")
        print(f"{driving_video.shape=}")
        params = {
            "source_image": source_image,
            "driving_video": driving_video,
            "generator": generator,
            "kp_detector": kp_detector,
            "relative_movement": relative_movement,
            "relative_jacobian": relative_jacobian,
            "adapt_movement_scale": adapt_movement_scale,
        }

        if find_best_frame:
            predictions = inference_best_frame(**params)
        else:
            predictions = inference(**params)

        output_images = out_video(predictions)

        return (
            output_images,
            audio,
            frame_rate,
        )


NODE_CLASS_MAPPINGS = {
    "FOMM_Runner": FOMM_Runner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FOMM_Runner": "FOMM Runner",
}
