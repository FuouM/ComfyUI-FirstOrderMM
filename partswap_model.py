import torch.nn.functional as F

from .modules.dense_motion import DenseMotionNetwork
from .modules.reconstruction_module import ReconstructionModule
from .modules.util import AntiAliasInterpolation2d


class PartSwapGenerator(ReconstructionModule):
    def __init__(self, blend_scale=1, first_order_motion_model=False, **kwargs):
        super(PartSwapGenerator, self).__init__(**kwargs)
        if blend_scale == 1:
            self.blend_downsample = lambda x: x
        else:
            self.blend_downsample = AntiAliasInterpolation2d(1, blend_scale)

        if first_order_motion_model:
            self.dense_motion_network = DenseMotionNetwork()
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
            motion = self.dense_motion_network(
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
