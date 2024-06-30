import torch
import numpy as np
from comfy.utils import ProgressBar
from tqdm import tqdm
from scipy.spatial import ConvexHull
from .modules.generator import OcclusionAwareGenerator
from .modules.keypoint_detector import KPDetector

from .utils import tensor_to_numpy

try:
    import face_alignment

    face_alignment_available = True
except ImportError:
    face_alignment_available = False
    print("Warning: face_alignment module not found. fine_best_frame will not work. ")


def normalize_kp_best_frame(kp):
    kp = kp - kp.mean(axis=0, keepdims=True)
    area = ConvexHull(kp[:, :2]).volume
    area = np.sqrt(area)
    kp[:, :2] = kp[:, :2] / area
    return kp

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

def find_best_frame(source_image, driving_video: list, cpu=False) -> int:
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