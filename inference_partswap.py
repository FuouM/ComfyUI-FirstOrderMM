import numpy as np
import torch
import torch.nn.functional as F
import yaml
from comfy.utils import ProgressBar
from tqdm import tqdm

from .modules.dense_motion import DenseMotionNetwork
from .modules.reconstruction_module import ReconstructionModule
from .modules.segmentation_module import SegmentationModule
from .modules.util import AntiAliasInterpolation2d
from .sync_batchnorm.replicate import DataParallelWithCallback


def face_parse_seg(
    source_image, face_parser_model, in_size=(512, 512), out_size=(64, 64)
):
    seg = F.interpolate(source_image, size=in_size)
    seg = (seg - face_parser_model.mean) / face_parser_model.std
    seg = torch.softmax(face_parser_model(seg)[0], dim=1)
    seg = F.interpolate(seg, size=out_size)
    return seg


class PartSwapGenerator(ReconstructionModule):
    def __init__(self, blend_scale=1, first_order_motion_model=False, **kwargs):
        super(PartSwapGenerator, self).__init__(**kwargs)
        if blend_scale == 1:
            self.blend_downsample = lambda x: x
        else:
            self.blend_downsample = AntiAliasInterpolation2d(1, blend_scale)

        if first_order_motion_model:
            self.dense_motion_network = DenseMotionNetwork(
                block_expansion=64,
                num_blocks=5,
                max_features=1024,
                num_kp=10,
                num_channels=3,
                estimate_occlusion_map=True,
                scale_factor=0.25,
            )
        else:
            self.dense_motion_network = None

    def forward(
        self,
        source_image,
        target_image,
        seg_target,
        seg_source,
        blend_mask,
        use_source_segmentation=False,
    ):
        # Encoding of source image
        enc_source = self.first(source_image)
        for i in range(len(self.down_blocks)):
            enc_source = self.down_blocks[i](enc_source)

        # Encoding of target image
        enc_target = self.first(target_image)
        for i in range(len(self.down_blocks)):
            enc_target = self.down_blocks[i](enc_target)

        output_dict = {}
        # Compute flow field for source image
        if self.dense_motion_network is None:
            segment_motions = self.segment_motion(seg_target, seg_source)
            segment_motions = segment_motions.permute(0, 1, 4, 2, 3)
            mask = seg_target["segmentation"].unsqueeze(2)
            deformation = (segment_motions * mask).sum(dim=1)
            deformation = deformation.permute(0, 2, 3, 1)
        else:
            motion = self.dense_motion_network.forward_partswap(
                source_image=source_image, seg_target=seg_target, seg_source=seg_source
            )
            deformation = motion["deformation"]

        # Deform source encoding according to the motion
        enc_source = self.deform_input(enc_source, deformation)

        if self.estimate_visibility:
            if self.dense_motion_network is None:
                visibility = seg_source["segmentation"][:, 1:].sum(
                    dim=1, keepdim=True
                ) * (
                    1
                    - seg_target["segmentation"][:, 1:]
                    .sum(dim=1, keepdim=True)
                    .detach()
                )
                visibility = 1 - visibility
            else:
                visibility = motion["visibility"]

            if (
                enc_source.shape[2] != visibility.shape[2]
                or enc_source.shape[3] != visibility.shape[3]
            ):
                visibility = F.interpolate(
                    visibility, size=enc_source.shape[2:], mode="bilinear"
                )
            enc_source = enc_source * visibility

        blend_mask = self.blend_downsample(blend_mask)
        # If source segmentation is provided use it should be deformed before blending
        if use_source_segmentation:
            blend_mask = self.deform_input(blend_mask, deformation)

        out = enc_target * (1 - blend_mask) + enc_source * blend_mask

        out = self.bottleneck(out)

        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)

        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict


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
            # print(f"{blend_mask.shape=}")
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


def partial_state_dict_load(module, state_dict):
    own_state = module.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue

        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


def load_reconstruction_module(module, checkpoint):
    if "generator" in checkpoint:
        partial_state_dict_load(module, checkpoint["generator"])
    else:
        module.load_state_dict(checkpoint["reconstruction_module"])


def load_segmentation_module(module, checkpoint):
    if "kp_detector" in checkpoint:
        partial_state_dict_load(module, checkpoint["kp_detector"])
        module.state_dict()["affine.weight"].copy_(
            checkpoint["kp_detector"]["jacobian.weight"]
        )
        module.state_dict()["affine.bias"].copy_(
            checkpoint["kp_detector"]["jacobian.bias"]
        )
        module.state_dict()["shift.weight"].copy_(
            checkpoint["kp_detector"]["kp.weight"]
        )
        module.state_dict()["shift.bias"].copy_(checkpoint["kp_detector"]["kp.bias"])
        if "semantic_seg.weight" in checkpoint["kp_detector"]:
            module.state_dict()["segmentation.weight"].copy_(
                checkpoint["kp_detector"]["semantic_seg.weight"]
            )
            module.state_dict()["segmentation.bias"].copy_(
                checkpoint["kp_detector"]["semantic_seg.bias"]
            )
        else:
            print("Segmentation part initialized at random.")
    else:
        module.load_state_dict(checkpoint["segmentation_module"])


def load_partswap_checkpoint(
    config_path: str,
    checkpoint_path: str,
    blend_scale: float,
    use_fomm: bool,
    cpu=False,
):
    with open(config_path) as f:
        config = yaml.full_load(f)

    reconstruction_module = PartSwapGenerator(
        blend_scale=blend_scale,
        first_order_motion_model=use_fomm,
        **config["model_params"]["reconstruction_module_params"],
        **config["model_params"]["common_params"],
    )

    segmentation_module = SegmentationModule(
        **config["model_params"]["segmentation_module_params"],
        **config["model_params"]["common_params"],
    )

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    else:
        segmentation_module.cuda()
        checkpoint = torch.load(checkpoint_path)

    load_reconstruction_module(reconstruction_module, checkpoint)
    load_segmentation_module(segmentation_module, checkpoint)

    if not cpu:
        reconstruction_module = DataParallelWithCallback(reconstruction_module)
        segmentation_module = DataParallelWithCallback(segmentation_module)

    reconstruction_module.eval()
    segmentation_module.eval()

    return reconstruction_module, segmentation_module
