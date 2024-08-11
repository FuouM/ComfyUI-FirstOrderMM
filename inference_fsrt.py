import numpy as np
import torch
import tqdm
import yaml
from comfy.utils import ProgressBar
from scipy.spatial import ConvexHull

from .inference_fomm import find_best_frame
from .module_fsrt.checkpoint import Checkpoint
from .module_fsrt.expression_encoder import ExpressionEncoder
from .module_fsrt.keypoint_detector import KPDetector
from .module_fsrt.model import FSRT


def fsrt_inference(
    source_image,
    driving_video: list,
    config_path: str,
    checkpoint_path: str,
    keypoint_path: str,
    relative=False,  # use relative or absolute keypoint coordinates
    adapt_scale=False,  # adapt movement scale based on convex hull of keypoints
    find_best_frame=False,  # Generate from the frame that is the most alligned with source
    max_num_pixels=65536,  # Number of parallel processed pixels. Reduce this value if you run out of GPU memory!
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path) as f:
        cfg = yaml.full_load(f)

    kp_detector = KPDetector().to(device)
    kp_detector.load_state_dict(torch.load(keypoint_path))
    expression_encoder = ExpressionEncoder(
        expression_size=cfg["model"]["expression_size"],
        in_channels=kp_detector.predictor.out_filters,
    )

    model = FSRT(cfg["model"], expression_encoder=expression_encoder).to(device)

    model.eval()
    kp_detector.eval()

    encoder_module = model.encoder
    decoder_module = model.decoder
    expression_encoder_module = model.expression_encoder

    checkpoint = Checkpoint(
        "./",
        device=device,
        encoder=encoder_module,
        decoder=decoder_module,
        expression_encoder=expression_encoder_module,
    )

    _ = checkpoint.load(checkpoint_path)

    source_image = source_image.to(device)
    if find_best_frame:
        predictions = inference_best_frame(
            source_image,
            driving_video,
            model,
            kp_detector,
            cfg,
            device,
            max_num_pixels,
            relative=relative,
            adapt_movement_scale=adapt_scale,
        )
    else:
        predictions = inference(
            source_image,
            driving_video,
            model,
            kp_detector,
            cfg,
            device,
            max_num_pixels,
            relative=relative,
            adapt_movement_scale=adapt_scale,
        )

    return predictions


def inference_best_frame(
    source_image,
    driving_video,
    model,
    kp_detector,
    cfg,
    device,
    max_num_pixels,
    relative=False,
    adapt_movement_scale=False,
):
    best_frame_idx = find_best_frame(source_image, driving_video)

    first_half = driving_video[
        :, :, : best_frame_idx + 1
    ]  # Include the best frame in the first half
    first_half = torch.flip(first_half, dims=[2])  # Reverse the first half
    second_half = driving_video[:, :, best_frame_idx + 1 :]

    predictions_first = inference(
        source_image,
        first_half,
        model,
        kp_detector,
        cfg,
        device,
        max_num_pixels,
        relative,
        adapt_movement_scale,
    )
    predictions_second = inference(
        source_image,
        second_half,
        model,
        kp_detector,
        cfg,
        device,
        max_num_pixels,
        relative,
        adapt_movement_scale,
    )

    predictions = []
    predictions_first = predictions_first[::-1]  # Reverse the first half back
    predictions.extend(predictions_first)
    predictions.extend(predictions_second)

    return predictions


def inference(
    source_image,
    driving_video,
    model,
    kp_detector,
    cfg,
    device,
    max_num_pixels,
    relative=False,
    adapt_movement_scale=False,
):
    source_image = source_image.permute(0, 2, 3, 1)

    _, y, x = np.meshgrid(
        np.zeros(2),
        np.arange(source_image.shape[-3]),
        np.arange(source_image.shape[-2]),
        indexing="ij",
    )
    idx_grids = np.stack([x, y], axis=-1).astype(np.float32)
    # Normalize
    idx_grids[..., 0] = (idx_grids[..., 0] + 0.5 - ((source_image.shape[-3]) / 2.0)) / (
        (source_image.shape[-3]) / 2.0
    )
    idx_grids[..., 1] = (idx_grids[..., 1] + 0.5 - ((source_image.shape[-2]) / 2.0)) / (
        (source_image.shape[-2]) / 2.0
    )
    idx_grids = torch.from_numpy(idx_grids).to(device).unsqueeze(0)
    z = None

    with torch.no_grad():
        predictions = []
        source = source_image.permute(0, 3, 1, 2)
        # driving = driving_video.permute(0, 4, 1, 2, 3)
        driving = driving_video
        # print(f"{source.shape=}")
        # print(f"{driving_video.shape=}")
        kp_source, expression_vector_src = extract_keypoints_and_expression(
            source.clone(), model, kp_detector, cfg, src=True
        )
        kp_driving_initial, _ = extract_keypoints_and_expression(
            driving[:, :, 0].to(device).clone(), model, kp_detector, cfg
        )

        num_frames = driving.shape[2]
        pbar = ProgressBar(num_frames)
        for frame_idx in tqdm.tqdm(range(num_frames), desc="Generating"):
            driving_frame = driving[:, :, frame_idx].to(device)
            kp_driving, expression_vector_driv = extract_keypoints_and_expression(
                driving_frame.clone(), model, kp_detector, cfg
            )
            kp_norm = normalize_kp(
                kp_source=kp_source[0],
                kp_driving=kp_driving,
                kp_driving_initial=kp_driving_initial,
                use_relative_movement=relative,
                adapt_movement_scale=adapt_movement_scale,
            )
            out, z = forward_model(
                model,
                expression_vector_src,
                kp_source,
                expression_vector_driv,
                kp_norm,
                source.unsqueeze(0),
                idx_grids,
                cfg,
                max_num_pixels,
                z=z,
            )

            pred = torch.clamp(out[0], 0.0, 1.0)
            predictions.append(pred.unsqueeze(0))
            pbar.update_absolute(frame_idx, num_frames)

    return predictions


def forward_model(
    model,
    expression_vector_src,
    keypoints_src,
    expression_vector_driv,
    keypoints_driv,
    img_src,
    idx_grids,
    cfg,
    max_num_pixels,
    z=None,
):
    # render_kwargs = cfg["model"]["decoder_kwargs"]
    if len(img_src.shape) < 5:
        img_src = img_src.unsqueeze(1)
    if len(keypoints_src.shape) < 4:
        keypoints_src = keypoints_src.unsqueeze(1)

    if z is None:
        z = model.encoder(
            img_src,
            keypoints_src,
            idx_grids[:, :1].repeat(1, img_src.shape[1], 1, 1, 1),
            expression_vector=expression_vector_src,
        )

    target_pos = idx_grids[:, 1]
    target_kps = keypoints_driv

    _, height, width = target_pos.shape[:3]
    target_pos = target_pos.flatten(1, 2)

    target_kps = target_kps.unsqueeze(1).repeat(1, target_pos.shape[1], 1, 1)

    num_pixels = target_pos.shape[1]
    img = torch.zeros((target_pos.shape[0], target_pos.shape[1], 3))

    for i in range(0, num_pixels, max_num_pixels):
        img[:, i : i + max_num_pixels], extras = model.decoder(
            z.clone(),
            target_pos[:, i : i + max_num_pixels],
            target_kps[:, i : i + max_num_pixels],
            expression_vector=expression_vector_driv,
        )

    return img.view(img.shape[0], height, width, 3), z


def normalize_kp(
    kp_source,
    kp_driving,
    kp_driving_initial,
    adapt_movement_scale=False,
    use_relative_movement=False,
):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source.data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial[0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = kp_driving

    if use_relative_movement:
        kp_value_diff = kp_driving - kp_driving_initial
        kp_value_diff *= adapt_movement_scale
        kp_new = kp_value_diff + kp_source

    return kp_new


def extract_keypoints_and_expression(img, model, kp_detector, cfg, src=False):
    assert kp_detector is not None

    bs, c, h, w = img.shape
    nkp = kp_detector.num_kp
    with torch.no_grad():
        kps, latent_dict = kp_detector(img)
        heatmaps = latent_dict["heatmap"].view(
            bs, nkp, latent_dict["heatmap"].shape[-2], latent_dict["heatmap"].shape[-1]
        )
        feature_maps = latent_dict["feature_map"].view(
            bs,
            latent_dict["feature_map"].shape[-3],
            latent_dict["feature_map"].shape[-2],
            latent_dict["feature_map"].shape[-1],
        )

    if kps.shape[1] == 1:
        kps = kps.squeeze(1)

    expression_vector = model.expression_encoder(feature_maps, heatmaps)

    if src:
        expression_vector = expression_vector[None]

    return kps, expression_vector
