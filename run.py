"""
@author: Fuou Marinas
@title: ComfyUI-FirstOrderMM
@nickname: FOMM
@description: ComfyUI-native nodes to run First Order Motion Model for Image Animation and its non-diffusion-based successors.
"""

from pathlib import Path

import torch

from .constants import (
    ARTICULATE_CFG_PATH,
    ARTICULATE_MODEL_PATH,
    SPLINE_CFG_PATH,
    SPLINE_DEFAULT,
    SPLINE_MODEL_PATH,
    SPLINE_MODES,
    config_folder,
    default_model_name,
    default_partswap_model_name,
    face_parser_checkpoint_name,
    partswap_fomm_model_names,
    partswap_model_config_dict,
    partswap_model_length_dict,
    partswap_model_names,
    supported_model_names,
)
from .face_parsing.face_parsing_loader import load_face_parser_model
from .inference_articulate import articulate_inference
from .inference_fomm import inference, inference_best_frame, load_checkpoint
from .inference_partswap import load_partswap_checkpoint, partswap_inference
from .inference_spline import spline_inference
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
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = (
        "IMAGE",
        "AUDIO",
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


class FOMM_Seg5Chooser:
    @classmethod
    def INPUT_TYPES(s):
        return build_seg_arguments("vox-5segments")

    RETURN_TYPES = ("STRING",)

    RETURN_NAMES = ("chosen_seg_indices",)

    FUNCTION = "todo"
    CATEGORY = "FirstOrderMM"

    def todo(self, **args):
        print(args)
        assert len(args) == 6, f"Expected 6 arguments, got {len(args)}"
        segments = list(args.values())
        seg_list = [i for i, seg in enumerate(segments) if seg]
        seg_list = serialize_integers(seg_list)
        return (seg_list,)


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


class FOMM_Seg15Chooser:
    @classmethod
    def INPUT_TYPES(s):
        return build_seg_arguments("vox-15segments")

    RETURN_TYPES = ("STRING",)

    RETURN_NAMES = ("chosen_seg_indices",)

    FUNCTION = "todo"
    CATEGORY = "FirstOrderMM"

    def todo(self, **args):
        print(args)
        assert len(args) == 16, f"Expected 16 arguments, got {len(args)}"
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
                    {"default": 1.0, "min": 0.6, "max": 1.0, "step": 0.01},
                ),
                "use_source_seg": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "hard_edges": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "use_face_parser": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "chosen_seg_indices": ("STRING", {}),
                "viz_alpha": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "AUDIO",
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
        use_face_parser: bool,
        chosen_seg_indices: str,
        viz_alpha: float,
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

        use_fomm = False
        if model_name in partswap_fomm_model_names:
            use_fomm = True

        reconstruction_module, segmentation_module = load_partswap_checkpoint(
            config_path, checkpoint_path, blend_scale, use_fomm=use_fomm
        )

        face_parser_model = None
        # if model_name in partswap_face_parser_enforce_models:
        #     assert use_face_parser, "Special supervised model, requires face_parser"
        if use_face_parser:
            face_parser_model = load_face_parser_model(
                base_dir, face_parser_checkpoint_name
            )
            if face_parser_model is not None:
                face_parser_model.cuda()
                face_parser_model.eval()

        print(f"Chosen {chosen_seg_indices}")
        swap_indices = deserialize_integers(chosen_seg_indices)
        swap_indices = [
            x for x in swap_indices if x < partswap_model_length_dict[model_name]
        ]
        print(f"Swapping {swap_indices}")

        params = {
            "swap_indices": swap_indices,
            "source_image": source_image,
            "target_video": driving_video,
            "reconstruction_module": reconstruction_module,
            "segmentation_module": segmentation_module,
            "use_source_seg": use_source_seg,
            "face_parser_model": face_parser_model,
            "hard_edges": hard_edges,
        }

        seg_source, seg_targets, predictions = partswap_inference(**params)

        seg_src_viz = visualize_frame(source_image, seg_source, alpha=viz_alpha)
        seg_tgt_viz = visualize_frame(
            driving_video_org[0], seg_targets[0], alpha=viz_alpha
        )

        output_images = out_video(predictions)

        del face_parser_model

        return (
            seg_src_viz,
            seg_tgt_viz,
            output_images,
            audio,
            frame_rate,
        )


class Articulate_Runner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "driving_video_input": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30.0}),
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = (
        "IMAGE",
        "AUDIO",
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
        frame_rate: float,
        audio=None,
    ):
        print(f"{source_image.shape=}")
        print(f"{driving_video_input.shape=}")
        print(f"{type(audio)=}")
        print(base_dir)

        config_path = f"{base_dir}/{ARTICULATE_CFG_PATH}"
        checkpoint_path = f"{base_dir}/{ARTICULATE_MODEL_PATH}"

        source_image = reshape_image(source_image, (256, 256))
        driving_video = reshape_image(driving_video_input, (256, 256)).unsqueeze(0)
        driving_video = driving_video.permute(0, 2, 1, 3, 4)

        print("After reshaping")
        print(f"{source_image.shape=}")
        print(f"{driving_video.shape=}")

        params = {
            "source_image": source_image,
            "driving_video": driving_video,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
        }

        predictions = articulate_inference(**params)

        output_images = out_video(predictions)

        return (
            output_images,
            audio,
            frame_rate,
        )


class Spline_Runner:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "driving_video_input": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 30.0}),
                "predict_mode": (SPLINE_MODES, {"default": SPLINE_DEFAULT}),
                "find_best_frame": (
                    "BOOLEAN",
                    {"default": False},
                ),  # Generate from the frame that is the most alligned with source
            },
            "optional": {"audio": ("AUDIO",)},
        }

    RETURN_TYPES = (
        "IMAGE",
        "AUDIO",
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
        frame_rate: float,
        predict_mode: str,
        find_best_frame: bool,
        audio=None,
    ):
        print(f"{type(source_image)=}")
        print(f"{type(driving_video_input)=}")
        print(f"{source_image.shape=}")
        print(f"{driving_video_input.shape=}")
        print(f"{type(audio)=}")
        print(base_dir)

        config_path = f"{base_dir}/{SPLINE_CFG_PATH}"
        checkpoint_path = f"{base_dir}/{SPLINE_MODEL_PATH}"

        source_image = reshape_image(source_image, (256, 256))
        driving_video = reshape_image(driving_video_input, (256, 256)).unsqueeze(0)
        driving_video = driving_video.permute(0, 2, 1, 3, 4)

        print("After reshaping")
        print(f"{source_image.shape=}")
        print(f"{driving_video.shape=}")
        params = {
            "source_image": source_image,
            "driving_video": driving_video,
            "config_path": config_path,
            "checkpoint_path": checkpoint_path,
            "predict_mode": predict_mode,
            "find_best_frame": find_best_frame,
        }

        predictions = spline_inference(**params)

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
