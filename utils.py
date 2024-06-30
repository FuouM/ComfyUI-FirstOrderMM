import numpy as np
import torch
import torch.nn.functional as F

from .constants import (
    checkpoint_folder,
    checkpoint_folder_link,
    checkpoint_suffix,
    config_folder,
    config_suffix,
    non_cpk_model_names,
    partswap_model_names,
    seg_model_dict,
    supported_model_names,
)


def check_supported(entity_name: str, entity_support_list: list[str]):
    assert (
        entity_name in entity_support_list
    ), f"Unknown {entity_name}, expected {entity_support_list}"


def get_model_file_name(model_name: str) -> str:
    check_supported(model_name, supported_model_names)

    if model_name in non_cpk_model_names:
        model_file_name = f"{checkpoint_folder}/{model_name}{checkpoint_suffix}"
    else:
        model_file_name = f"{checkpoint_folder}/{model_name}-cpk{checkpoint_suffix}"

    return model_file_name


def get_model_link(model_name: str) -> str:
    model_file_name = get_model_file_name(model_name)
    model_dl_link = f"{checkpoint_folder_link}/{model_file_name}"
    return model_dl_link


def get_config_path(model_name: str) -> str:
    config_path = f"{config_folder}/{model_name}-{config_suffix}"
    return config_path


def get_partswap_model_file_name(model_name: str):
    return f"{checkpoint_folder}/{model_name}{checkpoint_suffix}"


def out_video(predictions: list[np.ndarray]):
    out_tensor_list = []
    for i in predictions:
        out_img = torch.from_numpy(i)
        out_tensor_list.append(out_img)
    images = torch.stack(out_tensor_list, dim=0)
    return images


def reshape_image(image, target_size=(256, 256)):
    # Check if image is 4D (batch, height, width, channels)
    if image.dim() != 4:
        raise ValueError("Expected 4D input (batch, height, width, channels)")

    # Permute dimensions to [B, C, H, W] for resizing
    image = image.permute(0, 3, 1, 2)

    # Resize to target size
    resized_image = F.interpolate(
        image, size=target_size, mode="bilinear", align_corners=False
    )

    return resized_image


def tensor_to_numpy(tensor):
    # Convert [1, 3, H, W] tensor to [H, W, 3] numpy array
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def build_seg_arguments(model_name: str):
    check_supported(model_name, partswap_model_names)
    segment_names = seg_model_dict[model_name]
    required = dict()
    for segment_name in segment_names:
        default = True
        if segment_name.endswith("background"):
            default = False
        required[segment_name] = ("BOOLEAN", {"default": default})

    arguments = {"required": required}

    return arguments
