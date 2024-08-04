import torch
import numpy as np
from scipy.spatial import ConvexHull
import yaml
import tqdm
from comfy.utils import ProgressBar

from .fomm_inference import find_best_frame

from .spline_module.avd_network import AVDNetwork
from .spline_module.dense_motion import DenseMotionNetwork
from .spline_module.keypoint_detector import KPDetector
from .spline_module.inpainting_network import InpaintingNetwork


def spline_inference(
    source_image,
    driving_video: list,
    config_path: str,
    checkpoint_path: str,
    predict_mode="relative",
    find_best_frame=False,
    cpu=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(
        config_path, checkpoint_path, device
    )
    
    source_image = source_image.to(device)
    driving_video = driving_video.to(device)

    if find_best_frame:
        predictions = spline_inference_best_frame(
            source_image,
            driving_video,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            predict_mode,
            cpu,
        )
    else:
        predictions = inference(
            source_image,
            driving_video,
            inpainting,
            kp_detector,
            dense_motion_network,
            avd_network,
            predict_mode,
        )

    return predictions


def spline_inference_best_frame(
    source_image,
    driving_video: list,
    inpainting,
    kp_detector,
    dense_motion_network,
    avd_network,
    predict_mode,
    cpu=False,
):
    best_frame_idx = find_best_frame(source_image, driving_video, cpu)

    first_half = driving_video[
        :, :, : best_frame_idx + 1
    ]  # Include the best frame in the first half
    first_half = torch.flip(first_half, dims=[2])  # Reverse the first half
    second_half = driving_video[:, :, best_frame_idx + 1 :]

    predictions_first = inference(
        source_image,
        first_half,
        inpainting,
        kp_detector,
        dense_motion_network,
        avd_network,
        predict_mode,
    )

    predictions_second = inference(
        source_image,
        second_half,
        inpainting,
        kp_detector,
        dense_motion_network,
        avd_network,
        predict_mode,
    )

    predictions = []
    predictions_first = predictions_first[::-1]  # Reverse the first half back
    predictions.extend(predictions_first)
    predictions.extend(predictions_second)
    return predictions


def inference(
    source_image,
    driving_video: list,
    inpainting,
    kp_detector,
    dense_motion_network,
    avd_network,
    predict_mode,
):
    with torch.no_grad():
        source = source_image
        driving = driving_video
        predictions = []

        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        num_frames = driving.shape[2]
        pbar = ProgressBar(num_frames)

        for frame_idx in tqdm.tqdm(range(num_frames)):
            driving_frame = driving[:, :, frame_idx]
            kp_driving = kp_detector(driving_frame)
            if predict_mode == "standard":
                kp_norm = kp_driving
            elif predict_mode == "relative":
                kp_norm = relative_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                )
            elif predict_mode == "avd":
                kp_norm = avd_network(kp_source, kp_driving)

            dense_motion = dense_motion_network(
                source_image=source,
                kp_driving=kp_norm,
                kp_source=kp_source,
                bg_param=None,
                dropout_flag=False,
            )

            out = inpainting(source, dense_motion)
            out_np = np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
            predictions.append(out_np)
            pbar.update_absolute(frame_idx, num_frames)

    return predictions


def relative_kp(kp_source, kp_driving, kp_driving_initial):
    source_area = ConvexHull(kp_source["fg_kp"][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial["fg_kp"][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = kp_driving["fg_kp"] - kp_driving_initial["fg_kp"]
    kp_value_diff *= adapt_movement_scale
    kp_new["fg_kp"] = kp_value_diff + kp_source["fg_kp"]

    return kp_new


def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.full_load(f)

    inpainting = InpaintingNetwork(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"],
    )
    kp_detector = KPDetector(**config["model_params"]["common_params"])
    dense_motion_network = DenseMotionNetwork(
        **config["model_params"]["common_params"],
        **config["model_params"]["dense_motion_params"],
    )
    avd_network = AVDNetwork(
        num_tps=config["model_params"]["common_params"]["num_tps"],
        **config["model_params"]["avd_network_params"],
    )
    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)
    avd_network.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    inpainting.load_state_dict(checkpoint["inpainting_network"])
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    dense_motion_network.load_state_dict(checkpoint["dense_motion_network"])
    if "avd_network" in checkpoint:
        avd_network.load_state_dict(checkpoint["avd_network"])

    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()

    return inpainting, kp_detector, dense_motion_network, avd_network
