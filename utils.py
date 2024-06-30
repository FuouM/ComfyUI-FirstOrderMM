try:
    import face_alignment

    face_alignment_available = True
except ImportError:
    face_alignment_available = False
    print("Warning: face_alignment module not found. fine_best_frame will not work. ")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from comfy.utils import ProgressBar
from scipy.spatial import ConvexHull
from tqdm import tqdm

from .constants import (
    checkpoint_folder,
    checkpoint_folder_link,
    checkpoint_suffix,
    config_folder,
    non_cpk_model_names,
    supported_model_names,
)
from .modules.generator import OcclusionAwareGenerator
from .modules.keypoint_detector import KPDetector


def get_model_file_name(model_name: str) -> str:
    assert (
        model_name in supported_model_names
    ), f"Unknown model. Got {model_name}, expected {supported_model_names}"

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
    config_path = f"{config_folder}/{model_name}-256.yaml"
    return config_path


def out_video(predictions: list[np.ndarray]):
    out_tensor_list = []
    for i in predictions:
        # out_img = i.astype(np.float32) / 255.0
        out_img = torch.from_numpy(i)
        out_tensor_list.append(out_img)
    images = torch.stack(out_tensor_list, dim=0)
    return images


def load_checkpoint(
    config_path: str, checkpoint_path: str, cpu=False
) -> tuple[OcclusionAwareGenerator, KPDetector]:
    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"],
    )
    kp_detector = KPDetector(
        **config["model_params"]["kp_detector_params"],
        **config["model_params"]["common_params"],
    )

    if not cpu:
        generator.cuda()
        kp_detector.cuda()
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    generator.load_state_dict(checkpoint["generator"])
    kp_detector.load_state_dict(checkpoint["kp_detector"])

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector


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


def normalize_kp_best_frame(kp):
    kp = kp - kp.mean(axis=0, keepdims=True)
    area = ConvexHull(kp[:, :2]).volume
    area = np.sqrt(area)
    kp[:, :2] = kp[:, :2] / area
    return kp


def tensor_to_numpy(tensor):
    # Convert [1, 3, H, W] tensor to [H, W, 3] numpy array
    return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()


def find_best_frame(source_image, driving_video, cpu=False) -> int:
    if not face_alignment_available:
        print("Face Alignment module were not found. Best frame set to 0.")
        return 0
    print("Initializing Face Alignment...")
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=True,
        device="cpu" if cpu else "cuda",
    )
    print("Face alignment model loaded.")

    source_np = tensor_to_numpy(source_image)
    kp_source = fa.get_landmarks(255 * source_np)[0]
    kp_source = normalize_kp_best_frame(kp_source)

    norm = float("inf")
    frame_num = 0

    num_frames = driving_video.shape[2]

    for i in tqdm(range(num_frames)):
        # Extract single frame from driving video
        frame = driving_video[0, :, i]  # Shape: [3, 256, 256]
        frame_np = frame.permute(1, 2, 0).cpu().numpy()

        kp_driving = fa.get_landmarks(255 * frame_np)[0]
        kp_driving = normalize_kp_best_frame(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i

    print(f"Best frame: {frame_num}")

    return frame_num


def normalize_kp(
    kp_source,
    kp_driving,
    kp_driving_initial,
    adapt_movement_scale: bool,
    use_relative_movement: bool,
    use_relative_jacobian: bool,
):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source["value"][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(
            kp_driving_initial["value"][0].data.cpu().numpy()
        ).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1
    # print(f"{adapt_movement_scale=}")
    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = kp_driving["value"] - kp_driving_initial["value"]
        kp_value_diff *= adapt_movement_scale
        kp_new["value"] = kp_value_diff + kp_source["value"]

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"])
            )
            kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

    return kp_new


def inference(
    source_image,
    driving_video: list,
    generator: OcclusionAwareGenerator,
    kp_detector: KPDetector,
    relative_movement: bool,
    relative_jacobian: bool,
    adapt_movement_scale: bool,
    cpu=False,
) -> list:
    with torch.no_grad():
        source = source_image
        driving = driving_video
        predictions = []

        if not cpu:
            source = source.cuda()
            driving = driving.cuda()
            print("Using GPU")

        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])
        kp_norm = None

        num_frames = driving.shape[2]
        pbar = ProgressBar(num_frames)

        for frame_idx in tqdm(range(num_frames)):
            driving_frame = driving[:, :, frame_idx]
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(
                kp_source=kp_source,
                kp_driving=kp_driving,
                kp_driving_initial=kp_driving_initial,
                use_relative_movement=relative_movement,
                use_relative_jacobian=relative_jacobian,
                adapt_movement_scale=adapt_movement_scale,
            )
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            out_np = np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
            predictions.append(out_np)
            pbar.update_absolute(frame_idx, num_frames)

    return predictions


def inference_best_frame(
    source_image,
    driving_video,
    generator: OcclusionAwareGenerator,
    kp_detector: KPDetector,
    relative_movement: bool,
    relative_jacobian: bool,
    adapt_movement_scale: bool,
    cpu=False,
) -> list:
    best_frame_idx = find_best_frame(source_image, driving_video, cpu)

    first_half = driving_video[
        :, :, : best_frame_idx + 1
    ]  # Include the best frame in the first half
    first_half = torch.flip(first_half, dims=[2])  # Reverse the first half
    second_half = driving_video[:, :, best_frame_idx + 1 :]

    predictions_first = inference(
        source_image,
        first_half,
        generator,
        kp_detector,
        relative_movement,
        relative_jacobian,
        adapt_movement_scale,
        cpu,
    )
    predictions_second = inference(
        source_image,
        second_half,
        generator,
        kp_detector,
        relative_movement,
        relative_jacobian,
        adapt_movement_scale,
        cpu,
    )

    predictions = []
    predictions_first = predictions_first[::-1]  # Reverse the first half back
    predictions.extend(predictions_first)
    predictions.extend(predictions_second)
    return predictions
