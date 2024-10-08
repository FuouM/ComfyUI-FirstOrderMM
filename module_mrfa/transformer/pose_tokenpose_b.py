import torch
import torch.nn as nn

from .hr_base import HRNET_base
from .tokenpose_base import TokenPose_TB_base


class TokenPose_B(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(TokenPose_B, self).__init__()

        extra = cfg["MODEL"]["EXTRA"]
        self.cfg = cfg
        print(cfg["MODEL"])
        ##################################################
        self.pre_feature = HRNET_base(cfg, **kwargs)
        self.transformer = TokenPose_TB_base(
            feature_size=[cfg["MODEL"]["IMAGE_SIZE"][1] // 4, cfg["MODEL"]["IMAGE_SIZE"][0] // 4],
            patch_size=[cfg["MODEL"]["PATCH_SIZE"][1], cfg["MODEL"]["PATCH_SIZE"][0]],
            num_keypoints=cfg["MODEL"]["NUM_JOINTS"],
            dim=cfg["MODEL"]["DIM"],
            channels=cfg["MODEL"]["BASE_CHANNEL"],
            depth=cfg["MODEL"]["TRANSFORMER_DEPTH"],
            heads=cfg["MODEL"]["TRANSFORMER_HEADS"],
            mlp_dim=cfg["MODEL"]["DIM"] * cfg["MODEL"]["TRANSFORMER_MLP_RATIO"],
            apply_init=cfg["MODEL"]["INIT"],
            hidden_heatmap_dim=cfg["MODEL"]["HEATMAP_SIZE"][1]
            * cfg["MODEL"]["HEATMAP_SIZE"][0]
            // 8,
            heatmap_dim=cfg["MODEL"]["HEATMAP_SIZE"][1] * cfg["MODEL"]["HEATMAP_SIZE"][0],
            heatmap_size=[cfg["MODEL"]["HEATMAP_SIZE"][1], cfg["MODEL"]["HEATMAP_SIZE"][0]],
            pos_embedding_type=cfg["MODEL"]["POS_EMBEDDING_TYPE"],
            estimate_jacobian=cfg["MODEL"]["ESTIMATE_JACOBIAN"],
            temperature=cfg["MODEL"]["TEMPERATURE"],
            fix_img2motion_attention=cfg["MODEL"]["FIX_IMG2MOTION_ATTENTION"],
        )
        ###################################################

    def forward(self, x):
        if self.cfg["MODEL"]["DATA_PREPROCESS"]:
            mean = torch.tensor([0.485, 0.456, 0.406]).to(x.device)
            std = torch.tensor([0.229, 0.224, 0.225]).to(x.device)
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
            x = x - mean
            x = x / std
        x = self.pre_feature(x)
        x = self.transformer(x)
        return x

    def init_weights(self, pretrained=""):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    # cfg = convert_dict_to_attrit_dict(cfg)
    model = TokenPose_B(cfg, **kwargs)
    if is_train and cfg["MODEL"]["INIT_WEIGHTS"]:
        model.init_weights(cfg["MODEL"]["PRETRAINED"])

    return model
