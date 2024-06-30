import torch
import yaml

from .modules.segmentation_module import SegmentationModule
from .partswap_model import PartSwapGenerator
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
