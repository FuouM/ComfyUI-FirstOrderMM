import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm
import numpy as np

from .constants import (
    checkpoint_folder,
    checkpoint_suffix,
    config_folder,
    partswap_model_config_dict,
)
from .modules.dense_motion import DenseMotionNetwork
from .modules.reconstruction_module import ReconstructionModule
from .modules.segmentation_module import SegmentationModule
from .modules.util import AntiAliasInterpolation2d

from .sync_batchnorm.replicate import DataParallelWithCallback





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


def get_partswap_model_file_name(model_name: str):
    return f"{checkpoint_folder}/{model_name}{checkpoint_suffix}"


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
            
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

        return seg_source, seg_targets, predictions
    
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image 

def visualize_frame(image, seg_source, hard_edges=True, colormap='gist_rainbow', alpha=0.6):
    # Ensure image is the right shape and on CPU
    image = image.squeeze(0).cpu()  # Remove batch dimension
    
    # Extract segmentation from seg_source dictionary
    mask = seg_source["segmentation"]
    
    # Interpolate mask to match image size if necessary
    if mask.shape[2:] != image.shape[1:]:
        mask = F.interpolate(mask, size=image.shape[1:], mode='bilinear')
    
    if hard_edges:
        mask = (torch.max(mask, dim=1, keepdim=True)[0] == mask).float()
    
    colormap = plt.get_cmap(colormap)
    num_segments = mask.shape[1]
    mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
    color_mask = np.zeros((256, 256, 3), dtype=np.float32)
    patches = []
    for i in range(num_segments):
        if i != 0:
            color = np.array(colormap((i - 1) / (num_segments - 1)))[:3]
        else:
            color = np.array((0, 0, 0))
        patches.append(mpatches.Patch(color=color, label=str(i)))
        color_mask += mask[..., i:(i+1)] * color.reshape(1, 1, 3)
    
    # Convert image to numpy and normalize to [0, 1] range if necessary
    image_np = image.permute(1, 2, 0).numpy()
    if image_np.max() > 1:
        image_np = image_np / 255.0
    
    # Overlay the color mask on the original image
    overlaid_image = image_np * (1 - alpha) + color_mask * alpha
    
    # Ensure all values are in [0, 1] range
    overlaid_image = np.clip(overlaid_image, 0, 1)
    
    # Create a figure and axis
    fig = Figure(figsize=(10, 10))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    
    # Display the image
    ax.imshow(overlaid_image)
    ax.axis('off')
    
    # Add legend
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
    
    # Render the figure to numpy array
    fig.tight_layout(pad=0)
    canvas.draw()
    result_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    result_image = result_image.reshape(canvas.get_width_height()[::-1] + (3,))
    
    # # Resize to 256x256 if necessary
    # if result_image.shape[:2] != (256, 256):
    #     result_image = np.array(Image.fromarray(result_image).resize((256, 256)))
    
    # Convert to torch tensor and add batch dimension to get [1, 256, 256, 3]
    result_tensor = torch.from_numpy(result_image).unsqueeze(0).float() / 255.0
    
    return result_tensor