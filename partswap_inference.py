import numpy as np
import torch
from tqdm import tqdm

from comfy.utils import ProgressBar
from .modules.segmentation_module import SegmentationModule
from .partswap_model import PartSwapGenerator


def partswap_inference(
    swap_indices: list[int],
    source_image,
    target_video,
    reconstruction_module: PartSwapGenerator,
    segmentation_module: SegmentationModule,
    use_source_seg: bool,
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

        num_frames = target_video.shape[2]
        pbar = ProgressBar(num_frames)
        for frame_idx in tqdm(range(num_frames)):
            target_frame = target_video[:, :, frame_idx]
            seg_target = segmentation_module(target_frame)
            seg_targets.append(seg_target)
            if use_source_seg:
                blend_mask = seg_source["segmentation"]
            else:
                blend_mask = seg_target["segmentation"]

            blend_mask = blend_mask[:, swap_indices].sum(dim=1, keepdim=True)
            if hard_edges:
                blend_mask = (blend_mask > 0.5).type(blend_mask.type())

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
