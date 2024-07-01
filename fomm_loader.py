import yaml
import torch

from .modules.generator import OcclusionAwareGenerator
from .modules.keypoint_detector import KPDetector

def load_checkpoint(
    config_path: str, checkpoint_path: str, cpu=False
) -> tuple[OcclusionAwareGenerator, KPDetector]:
    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"],
    )
    kp_detector = KPDetector(
        **config["model_params"]["kp_detector_params"],
        **config["model_params"]["common_params"],
    )

    if not cpu:
        generator.cuda()
        kp_detector.cuda()
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    generator.load_state_dict(checkpoint["generator"])
    kp_detector.load_state_dict(checkpoint["kp_detector"])

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector