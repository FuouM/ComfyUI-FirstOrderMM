"""
@author: Fuou Marinas
@title: ComfyUI-FirstOrderMM
@nickname: FOMM
@description: Run First Order Motion Model for Image Animation in ComfyUI.
"""

from pathlib import Path
import torch

from .fomm_inference import inference, inference_best_frame
from .fomm_loader import load_checkpoint
from .partswap_inference import partswap_inference
from .partswap_loader import load_partswap_checkpoint

from .constants import (
    config_folder,
    default_model_name,
    default_partswap_model_name,
    partswap_model_config_dict,
    partswap_model_names,
    supported_model_names,
)
from .seg_viz import visualize_frame
from .utils import (
    build_seg_arguments,
    get_config_path,
    get_model_file_name,
    get_partswap_model_file_name,
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
        source_image,
        driving_video_input,
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


def serialize_integers(int_list):
    return "_".join(map(str, int_list))


def deserialize_integers(int_string):
    return list(map(int, int_string.split("_")))


class FOMM_Seg10Chooser:
    @classmethod
    def INPUT_TYPES(s):
        return build_seg_arguments("vox-10segments")

    RETURN_TYPES = ("STRING",)

    RETURN_NAMES = ("chosen_seg_indices",)

    FUNCTION = "todo"
    CATEGORY = "FirstOrderMM"

    def todo(self, **args):
        print(args)
        assert len(args) == 11, f"Expected 11 arguments, got {len(args)}"
        segments = list(args.values())
        seg_list = [i for i, seg in enumerate(segments) if seg]
        seg_list = serialize_integers(seg_list)
        return (seg_list,)


class FOMM_Partswap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "driving_video_input": ("IMAGE",),
                "model_name": (
                    partswap_model_names,
                    {"default": default_partswap_model_name},
                ),
                "frame_rate": ("FLOAT", {"default": 30.0}),
                "blend_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "use_source_seg": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "hard_edges": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "chosen_seg_indices": ("STRING", {}),
            },
            "optional": {"audio": ("VHS_AUDIO",)},
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "VHS_AUDIO",
        "FLOAT",
    )
    RETURN_NAMES = (
        "seg_src_viz",
        "seg_tgt_viz",
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
        blend_scale: float,
        use_source_seg: bool,
        hard_edges: bool,
        chosen_seg_indices: str,
        audio=None,
    ):
        print(f"{type(source_image)=}")
        print(f"{type(driving_video_input)=}")
        print(f"{source_image.shape=}")
        print(f"{driving_video_input.shape=}")
        print(f"{type(audio)=}")
        print(base_dir)

        config_path = (
            f"{base_dir}/{config_folder}/{partswap_model_config_dict[model_name]}"
        )
        checkpoint_path = f"{base_dir}/{get_partswap_model_file_name(model_name)}"

        source_image = reshape_image(source_image, (256, 256))
        driving_video_org = reshape_image(driving_video_input, (256, 256))
        driving_video = driving_video_org.unsqueeze(0).permute(0, 2, 1, 3, 4)

        reconstruction_module, segmentation_module = load_partswap_checkpoint(
            config_path, checkpoint_path, blend_scale, use_fomm=False
        )
        print(f"Chosen {chosen_seg_indices}")
        swap_indices = deserialize_integers(chosen_seg_indices)
        print(f"Swapping {swap_indices}")

        params = {
            "swap_indices": swap_indices,
            "source_image": source_image,
            "target_video": driving_video,
            "reconstruction_module": reconstruction_module,
            "segmentation_module": segmentation_module,
            "use_source_seg": use_source_seg,
            "hard_edges": hard_edges,
        }

        seg_source, seg_targets, predictions = partswap_inference(**params)

        seg_src_viz = visualize_frame(source_image, seg_source)
        seg_tgt_viz = visualize_frame(driving_video_org[0], seg_targets[0])

        output_images = out_video(predictions)

        return (
            seg_src_viz,
            seg_tgt_viz,
            output_images,
            audio,
            frame_rate,
        )


# NODE_CLASS_MAPPINGS = {
#     "FOMM_Runner": FOMM_Runner,
#     "FOMM_Partswap": FOMM_Partswap,
#     "FOMM_Seg10Chooser": FOMM_Seg10Chooser,
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "FOMM_Runner": "FOMM Runner",
#     "FOMM_Partswap": "FOMM Partswap",
#     "FOMM_Seg10Chooser": "FOMM_Seg10Chooser",
# }
