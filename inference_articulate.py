import numpy as np
import torch
import tqdm
import yaml
from comfy.utils import ProgressBar
from scipy.spatial import ConvexHull

from .articulate_module.avd_network import AVDNetwork
from .articulate_module.bg_motion_predictor import BGMotionPredictor
from .articulate_module.generator import Generator
from .articulate_module.region_predictor import RegionPredictor
from .sync_batchnorm.replicate import DataParallelWithCallback


def articulate_inference(
    source_image,
    driving_video: list,
    config_path: str,
    checkpoint_path: str,
    estimate_affine=True,
    pca_based=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path) as f:
        config = yaml.full_load(f)

    generator, region_predictor, bg_predictor, avd_network = init_models(
        config, estimate_affine, pca_based
    )
    generator = generator.to(device)
    region_predictor = region_predictor.to(device)
    bg_predictor = bg_predictor.to(device)
    avd_network = avd_network.to(device)

    animate_params = config["animate_params"]

    load_cpk(checkpoint_path, generator, region_predictor, bg_predictor, avd_network)

    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        region_predictor = DataParallelWithCallback(region_predictor)
        avd_network = DataParallelWithCallback(avd_network)

    generator.eval()
    region_predictor.eval()
    avd_network.eval()

    source_frame = source_image
    driving = driving_video
    predictions = []

    num_frames = driving.shape[2]
    pbar = ProgressBar(num_frames)

    with torch.no_grad():
        source_region_params = region_predictor(source_frame)
        driving_region_params_initial = region_predictor(driving_video[:, :, 0])

        for frame_idx in tqdm.tqdm(range(num_frames)):
            driving_frame = driving[:, :, frame_idx]
            driving_region_params = region_predictor(driving_frame)
            new_region_params = get_animation_region_params(
                source_region_params,
                driving_region_params,
                driving_region_params_initial,
                mode=animate_params["mode"],
                avd_network=avd_network,
            )
            out = generator(
                source_frame,
                source_region_params=source_region_params,
                driving_region_params=new_region_params,
            )

            out["driving_region_params"] = driving_region_params
            out["source_region_params"] = source_region_params
            out["new_region_params"] = new_region_params

            # visualization = Visualizer(**config["visualizer_params"]).visualize(
            #     source=source_frame, driving=driving_frame, out=out
            # ) / 255.0

            prediction = out["prediction"].data.cpu().numpy()
            prediction = np.transpose(prediction, [0, 2, 3, 1]).squeeze(0)

            # visualizations.append(visualization)
            predictions.append(prediction)

            pbar.update_absolute(frame_idx, num_frames)

    # print(f"{predictions[0].shape=}")
    # print(f"{visualizations[0].shape=}")

    # return predictions, visualizations
    return predictions


def init_models(config, estimate_affine, pca_based):
    generator = Generator(
        num_regions=config["model_params"]["num_regions"],
        num_channels=config["model_params"]["num_channels"],
        revert_axis_swap=config["model_params"]["revert_axis_swap"],
        **config["model_params"]["generator_params"],
    )

    config["model_params"]["region_predictor_params"]["pca_based"] = pca_based

    region_predictor = RegionPredictor(
        num_regions=config["model_params"]["num_regions"],
        num_channels=config["model_params"]["num_channels"],
        estimate_affine=estimate_affine,
        **config["model_params"]["region_predictor_params"],
    )

    bg_predictor = BGMotionPredictor(
        num_channels=config["model_params"]["num_channels"],
        **config["model_params"]["bg_predictor_params"],
    )

    avd_network = AVDNetwork(
        num_regions=config["model_params"]["num_regions"],
        **config["model_params"]["avd_network_params"],
    )

    return generator, region_predictor, bg_predictor, avd_network


def load_cpk(
    checkpoint_path,
    generator=None,
    region_predictor=None,
    bg_predictor=None,
    avd_network=None,
    optimizer_reconstruction=None,
    optimizer_avd=None,
):
    checkpoint = torch.load(checkpoint_path)
    # print(checkpoint.keys())
    if generator is not None:
        generator.load_state_dict(checkpoint["generator"])
    if region_predictor is not None:
        region_predictor.load_state_dict(checkpoint["region_predictor"])
    if bg_predictor is not None:
        bg_predictor.load_state_dict(checkpoint["bg_predictor"])
    if avd_network is not None:
        if "avd_network" in checkpoint:
            avd_network.load_state_dict(checkpoint["avd_network"])

    if optimizer_reconstruction is not None:
        optimizer_reconstruction.load_state_dict(checkpoint["optimizer_reconstruction"])
        return checkpoint["epoch_reconstruction"]

    if optimizer_avd is not None:
        if "optimizer_avd" in checkpoint:
            optimizer_avd.load_state_dict(checkpoint["optimizer_avd"])
            return checkpoint["epoch_avd"]
        return 0

    return 0


def get_animation_region_params(
    source_region_params,
    driving_region_params,
    driving_region_params_initial,
    mode="standard",
    avd_network=None,
    adapt_movement_scale=True,
):
    assert mode in ["standard", "relative", "avd"]
    new_region_params = {k: v for k, v in driving_region_params.items()}
    if mode == "standard":
        return new_region_params
    elif mode == "relative":
        source_area = ConvexHull(
            source_region_params["shift"][0].data.cpu().numpy()
        ).volume
        driving_area = ConvexHull(
            driving_region_params_initial["shift"][0].data.cpu().numpy()
        ).volume
        movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

        shift_diff = (
            driving_region_params["shift"] - driving_region_params_initial["shift"]
        )
        shift_diff *= movement_scale
        new_region_params["shift"] = shift_diff + source_region_params["shift"]

        affine_diff = torch.matmul(
            driving_region_params["affine"],
            torch.inverse(driving_region_params_initial["affine"]),
        )
        new_region_params["affine"] = torch.matmul(
            affine_diff, source_region_params["affine"]
        )
        return new_region_params
    elif mode == "avd":
        new_region_params = avd_network(source_region_params, driving_region_params)
        return new_region_params
