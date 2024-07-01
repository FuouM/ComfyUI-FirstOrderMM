import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from comfy.utils import ProgressBar
from .modules.segmentation_module import SegmentationModule
from .partswap_model import PartSwapGenerator


def face_parse_seg(
    source_image, face_parser_model, in_size=(512, 512), out_size=(64, 64)
):
    seg = F.interpolate(source_image, size=in_size)
    seg = (seg - face_parser_model.mean) / face_parser_model.std
    seg = torch.softmax(face_parser_model(seg)[0], dim=1)
    seg = F.interpolate(seg, size=out_size)
    return seg


def partswap_inference(
    swap_indices: list[int],
    source_image,
    target_video,
    reconstruction_module: PartSwapGenerator,
    segmentation_module: SegmentationModule,
    use_source_seg: bool,
    face_parser_model=None,
    hard_edges=False,
    cpu=False,
) -> list:
    with torch.no_grad():
        predictions = []
        seg_targets = []
        if not cpu:
            source_image = source_image.cuda()
            target_video = target_video.cuda()
            print("Using GPU")

        seg_source = segmentation_module(source_image)
        # print(f"{seg_source['segmentation'].shape}")

        num_frames = target_video.shape[2]
        pbar = ProgressBar(num_frames)
        for frame_idx in tqdm(range(num_frames)):
            target_frame = target_video[:, :, frame_idx]
            seg_target = segmentation_module(target_frame)
            seg_targets.append(seg_target)

            if face_parser_model is not None:
                blend_mask = face_parse_seg(
                    source_image if use_source_seg else target_frame,
                    face_parser_model,
                    (512, 512),
                    (64, 64),
                )

            else:
                blend_mask = (
                    seg_source["segmentation"]
                    if use_source_seg
                    else seg_target["segmentation"]
                )

            blend_mask = blend_mask[:, swap_indices].sum(dim=1, keepdim=True)
            if hard_edges:
                blend_mask = (blend_mask > 0.5).type(blend_mask.type())
            print(f"{blend_mask.shape=}")
            out = reconstruction_module(
                source_image,
                target_frame,
                seg_source=seg_source,
                seg_target=seg_target,
                blend_mask=blend_mask,
                use_source_segmentation=use_source_seg,
            )

            predictions.append(
                np.transpose(out["prediction"].data.cpu().numpy(), [0, 2, 3, 1])[0]
            )

            pbar.update_absolute(frame_idx, num_frames)

        return seg_source, seg_targets, predictions
