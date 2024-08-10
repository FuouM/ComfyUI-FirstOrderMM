import numpy as np
import torch
import tqdm
import yaml
from comfy.utils import ProgressBar
from scipy.spatial import ConvexHull

from .module_mrfa.model import MRFA
from .module_mrfa.util import AntiAliasInterpolation2d


def mrfa_inference(
    source_image: torch.Tensor,
    driving_video: torch.Tensor,
    config_path: str,
    checkpoint_path: str,
    use_relative=True,
    relative_movement=True,
    relative_jacobian=True,
    adapt_movement_scale=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path) as f:
        config = yaml.full_load(f)

    model = MRFA(config)
    model = torch.nn.DataParallel(model)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.module
    model.eval().to(device)

    kp_detector = model.encoder
    dense_motion_network = model.dense_motion
    decoder = model.decoder
    down = AntiAliasInterpolation2d(3, 0.25).to(device)

    source = source_image.to(device)  # 4D
    driving = driving_video.to(device)  #

    num_frames = driving.shape[2]
    pbar = ProgressBar(num_frames)

    predictions = []
    with torch.no_grad():
        # visualizations = []

        kp_source = kp_detector.forward(source)
        kp_driving_initial = kp_detector.forward(driving[:, :, 0])

        for frame_idx in tqdm.tqdm(
            range(num_frames),
            desc=f"Generating {'Relative' if use_relative else 'Absolute'} mode",
        ):
            driving_frame = driving[:, :, frame_idx]
            if use_relative:
                kp_driving = kp_detector.forward(driving_frame)
                kp_norm = normalize_kp(
                    kp_source=kp_source,
                    kp_driving=kp_driving,
                    kp_driving_initial=kp_driving_initial,
                    use_relative_movement=relative_movement,
                    use_relative_jacobian=relative_jacobian,
                    adapt_movement_scale=adapt_movement_scale,
                )
                dense_motion = dense_motion_network.forward(
                    source, kp_norm, kp_source, bg_param=None
                )

                if config["train_params"]["prior_model"] == "tpsm":
                    kp_s_value = kp_source["kp"].view(source.shape[0], -1, 5, 2).mean(2)
                    kp_d_value = kp_norm["kp"].view(driving.shape[0], -1, 5, 2).mean(2)
                else:
                    kp_s_value = kp_source["kp"]
                    kp_d_value = kp_norm["kp"]

                out, _, _ = decoder.forward(
                    kp_s_value,
                    kp_d_value,
                    dense_motion,
                    img=down(source),
                    img_full=source,
                )

                # visualization = Visualizer(**config['visualizer_params']).visualize(source=source, driving=driving_frame, out=out)
                # visualizations.append(visualization)

            else:
                abs_inp = {"source": source, "driving": driving_frame}
                out, _, _, _, _ = model.forward(abs_inp, is_train=False)

            prediction = out.data.cpu().numpy()
            prediction = np.transpose(prediction, [0, 2, 3, 1]).squeeze(0)
            predictions.append(prediction)

            pbar.update_absolute(frame_idx, num_frames)

    return predictions


def normalize_kp(
    kp_source,
    kp_driving,
    kp_driving_initial,
    adapt_movement_scale=False,
    use_relative_movement=False,
    use_relative_jacobian=False,
):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source["kp"][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial["kp"][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = kp_driving["kp"] - kp_driving_initial["kp"]
        kp_value_diff *= adapt_movement_scale
        kp_new["kp"] = kp_value_diff + kp_source["kp"]

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(
                kp_driving["jacobian"], torch.inverse(kp_driving_initial["jacobian"])
            )
            kp_new["jacobian"] = torch.matmul(jacobian_diff, kp_source["jacobian"])

    return kp_new
