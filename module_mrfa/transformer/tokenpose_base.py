import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
from torch import nn

from ..util import make_coordinate_grid

MIN_NUM_PATCHES = 16
BN_MOMENTUM = 0.1


class Residual(nn.Module):
    def __init__(self, fn, num_keypoints=10):
        super().__init__()
        self.fn = fn
        self.num_keypoints = num_keypoints

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class QKVPreNorm(nn.Module):
    def __init__(self, dim, fn, fusion_factor=1):
        super().__init__()
        self.norm = nn.LayerNorm(dim * fusion_factor)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        return self.fn(self.norm(q), self.norm(k), self.norm(v), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dropout=0.0,
        num_keypoints=None,
        scale_with_head=False,
        fix_img2motion_attention=False,
        num_img_tokens=None,
    ):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.num_keypoints = num_keypoints
        self.fix_img2motion_attention = fix_img2motion_attention
        if fix_img2motion_attention:
            num_tokens = num_keypoints + num_img_tokens
            mask = torch.zeros(num_tokens, num_tokens)
            mask[self.num_keypoints :, 0 : self.num_keypoints] += -1000.0
            mask = mask.unsqueeze(0).unsqueeze(0)
            self.register_buffer("attn_mask", mask)

    # @get_local('dots')
    # @get_local('attn')
    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.fix_img2motion_attention:
            dots += self.attn_mask
        attn = dots.softmax(dim=-1)
        # print(attn[0,0,-1,-1])
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class QKVAttention(nn.Module):
    def __init__(
        self, dim, heads=8, dropout=0.0, num_keypoints=None, scale_with_head=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim**-0.5

        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.num_keypoints = num_keypoints

    # @get_local('dots')
    # @get_local('attn')
    def forward(self, x, k=None, v=None, mask=None):
        b, n, _, h = *x.shape, self.heads
        q = self.to_q(x)
        k = self.to_k(k)
        v = self.to_v(v)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], "mask has incorrect dimensions"
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask
        attn = dots.softmax(dim=-1)
        # print(attn[0,0,-1,-1])
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout,
        num_keypoints=None,
        all_attn=False,
        scale_with_head=False,
        fix_img2motion_attention=False,
        num_patches=256,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        self.fix_img2motion_attention = fix_img2motion_attention
        for d in range(depth):
            # if not d < depth // 2:
            #     fix_img2motion_attention = False
            self.layers.append(
                nn.ModuleList(
                    [
                        Residual(
                            PreNorm(
                                dim,
                                Attention(
                                    dim,
                                    heads=heads,
                                    dropout=dropout,
                                    num_keypoints=num_keypoints,
                                    scale_with_head=scale_with_head,
                                    fix_img2motion_attention=fix_img2motion_attention,
                                    num_img_tokens=num_patches,
                                ),
                            ),
                            num_keypoints=self.num_keypoints,
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )

    def forward(self, x, mask=None, pos=None):
        img_token = x[:, self.num_keypoints :]
        for idx, (attn, ff) in enumerate(self.layers):
            if idx > 0 and self.all_attn:
                x[:, self.num_keypoints :] += pos
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_dim,
        dropout,
        num_keypoints=None,
        all_attn=False,
        scale_with_head=False,
        num_patches=256,
        v_pos=False,
    ):
        super().__init__()
        self.encoder = nn.ModuleList([])
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        self.v_pos = v_pos
        for d in range(depth):
            self.encoder.append(
                nn.ModuleList(
                    [
                        Residual(
                            QKVPreNorm(
                                dim,
                                QKVAttention(
                                    dim,
                                    heads=heads,
                                    dropout=dropout,
                                    num_keypoints=num_keypoints,
                                    scale_with_head=scale_with_head,
                                ),
                            ),
                            num_keypoints=self.num_keypoints,
                        ),
                        Residual(
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                        ),
                    ]
                )
            )

    def forward(self, x, mask=None, pos=None):
        for idx, (attn, ff) in enumerate(self.encoder):
            if self.all_attn:
                if self.v_pos:
                    x[:, self.num_keypoints :] += pos
                    q = x
                    k = x
                    v = x
                else:
                    v = x.clone()
                    x[:, self.num_keypoints :] += pos
                    q = x
                    k = x
            x = attn(q, k=k, v=v, mask=mask)
            x = ff(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TokenPose_TB_base(nn.Module):
    def __init__(
        self,
        *,
        feature_size,
        patch_size,
        num_keypoints,
        dim,
        depth,
        heads,
        mlp_dim,
        apply_init=False,
        apply_multi=True,
        hidden_heatmap_dim=64 * 6,
        heatmap_dim=64 * 64,
        heatmap_size=[64, 64],
        channels=3,
        dropout=0.0,
        emb_dropout=0.0,
        pos_embedding_type="sine-full",
        estimate_jacobian=True,
        temperature=0.1,
        spatial_kp_head=False,
        jacobian_token=True,
        hidden_dim=False,
        affine_jacobian=False,
        fix_img2motion_attention=False,
    ):
        super().__init__()
        assert isinstance(feature_size, list) and isinstance(
            patch_size, list
        ), "image_size and patch_size should be list"
        assert (
            feature_size[0] % patch_size[0] == 0
            and feature_size[1] % patch_size[1] == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (feature_size[0] // (patch_size[0])) * (
            feature_size[1] // (patch_size[1])
        )
        patch_dim = channels * patch_size[0] * patch_size[1]
        # assert pos_embedding_type in ['sine','learnable','sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = self.pos_embedding_type == "sine-full"

        self.jacobian_token = jacobian_token
        if jacobian_token:
            ## additional token for jacobian
            num_keypoints = 2 * num_keypoints
            self.num_keypoints = num_keypoints
        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h, w = (
            feature_size[0] // (self.patch_size[0]),
            feature_size[1] // (self.patch_size[1]),
        )
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            num_keypoints=num_keypoints,
            all_attn=self.all_attn,
            scale_with_head=True,
            fix_img2motion_attention=fix_img2motion_attention,
            num_patches=num_patches,
        )

        self.to_keypoint_token = nn.Identity()
        ## original mlphead for keypoint heatmap prediction
        self.spatial_kp_head = spatial_kp_head
        if spatial_kp_head:
            self.mlp_head = (
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, hidden_heatmap_dim),
                    nn.LayerNorm(hidden_heatmap_dim),
                    nn.Linear(hidden_heatmap_dim, heatmap_dim),
                )
                if (dim <= hidden_heatmap_dim * 0.5 and apply_multi)
                else nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, heatmap_dim))
            )
            self.temperature = temperature
        ## original mlphead for keypoint heatmap prediction
        else:
            if hidden_dim:
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.Linear(dim, 2),
                )
            else:
                self.mlp_head = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, 2),
                )
        trunc_normal_(self.keypoint_token, std=0.02)
        if apply_init:
            self.apply(self._init_weights)

        ## light-weight head for direct jacobian regression
        if estimate_jacobian:
            if hidden_dim:
                self.mlp_head_jacobian = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.Linear(dim, 4),
                )
                self._init_weights(self.mlp_head_jacobian[0])
                self._init_weights(self.mlp_head_jacobian[1])
                self._init_weights(self.mlp_head_jacobian[2])
            else:
                self.mlp_head_jacobian = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, 4),
                )
                self._init_weights(self.mlp_head_jacobian[0])
            self.mlp_head_jacobian[-1].weight.data.zero_()
            self.mlp_head_jacobian[-1].bias.data.copy_(
                torch.tensor([1, 0, 0, 1], dtype=torch.float)
            )
            self.affine_jacobian = affine_jacobian
        ## light-weight head for direct keypoint and jacobian regression

    def _make_position_embedding(self, w, h, d_model, pe_type="sine"):
        """
        d_model: embedding size in transformer encoder
        """
        assert pe_type in ["none", "learnable", "sine", "sine-full"]
        if pe_type == "none":
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == "learnable":
                self.pos_embedding = nn.Parameter(
                    torch.zeros(1, self.num_patches + self.num_keypoints, d_model)
                )
                trunc_normal_(self.pos_embedding, std=0.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model), requires_grad=False
                )
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(
        self, d_model, temperature=10000, scale=2 * math.pi
    ):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        # print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = (
            make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        )
        value = (heatmap * grid).sum(dim=(2, 3))  # N * 10 * 2
        kp = {"value": value}

        return kp

    def forward(self, feature, mask=None):
        p = self.patch_size
        # transformer
        x = rearrange(
            feature, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p[0], p2=p[1]
        )
        # print(x.shape)
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, "() n d -> b n d", b=b)
        if self.pos_embedding_type in ["sine", "sine-full"]:
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        elif self.pos_embedding_type == "none":
            x = torch.cat((keypoint_tokens, x), dim=1)
        else:
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, : (n + self.num_keypoints)]
        x = self.dropout(x)

        x = self.transformer(x, mask, self.pos_embedding)
        if self.jacobian_token:
            x_keypoint = self.to_keypoint_token(x[:, 0 : self.num_keypoints // 2])
        else:
            x_keypoint = self.to_keypoint_token(x[:, 0 : self.num_keypoints])
        if self.spatial_kp_head:
            heatmap = self.mlp_head(x_keypoint)
            heatmap = F.softmax(heatmap / self.temperature, dim=2)
            heatmap = rearrange(
                heatmap,
                "b c (p1 p2) -> b c p1 p2",
                p1=self.heatmap_size[0],
                p2=self.heatmap_size[1],
            )
            out = self.gaussian2kp(heatmap)
            out["heatmap"] = heatmap
        else:
            # keypoint = torch.tanh(self.mlp_head(x_keypoint))
            keypoint = 2 * F.sigmoid(self.mlp_head(x_keypoint)) - 1
            out = {"kp": keypoint}
        if self.mlp_head_jacobian is not None:
            if self.jacobian_token:
                x_jacobian = self.to_keypoint_token(
                    x[:, self.num_keypoints // 2 : self.num_keypoints]
                )
            else:
                x_jacobian = x_keypoint
            jacobian = self.mlp_head_jacobian(x_jacobian)
            if self.affine_jacobian:
                theta = jacobian[:, :, 0:2]
                theta = theta / (torch.norm(theta, p=2, dim=-1, keepdim=True) + 1e-10)
                cos_theta = theta[:, :, 0:1]
                sin_theta = theta[:, :, 1:2]
                rotate = torch.cat(
                    (cos_theta, -sin_theta, sin_theta, cos_theta), dim=-1
                )
                rotate = rearrange(rotate, "b c (p1 p2) -> b c p1 p2", p1=2, p2=2)

                scale = jacobian[:, :, 2:]
                # scale = torch.abs(jacobian[:,:,2:]) + 0.1
                scale = torch.tanh(scale) * 0.9 + 1
                scale = 1 / scale
                scale_x = scale[:, :, 0:1]
                scale_y = scale[:, :, 1:2]
                scale = torch.cat(
                    (
                        scale_x,
                        torch.zeros(scale_x.shape).to(x.device),
                        torch.zeros(scale_x.shape).to(x.device),
                        scale_y,
                    ),
                    dim=-1,
                )
                scale = rearrange(scale, "b c (p1 p2) -> b c p1 p2", p1=2, p2=2)

                jacobian = torch.matmul(rotate, scale)
            else:
                jacobian = rearrange(jacobian, "b c (p1 p2) -> b c p1 p2", p1=2, p2=2)
            out["jacobian"] = jacobian
        return out


class TokenPose_S_base(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_keypoints,
        dim,
        depth,
        heads,
        mlp_dim,
        apply_init=False,
        apply_multi=True,
        hidden_heatmap_dim=64 * 6,
        heatmap_dim=64 * 48,
        heatmap_size=[64, 48],
        channels=3,
        dropout=0.0,
        emb_dropout=0.0,
        pos_embedding_type="learnable",
        estimate_jacobian=False,
        temperature=0.1,
        jacobian_token=True,
        hidden_dim=False,
        affine_jacobian=False,
    ):
        super().__init__()
        assert isinstance(image_size, list) and isinstance(
            patch_size, list
        ), "image_size and patch_size should be list"
        assert (
            image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size[0] // (4 * patch_size[0])) * (
            image_size[1] // (4 * patch_size[1])
        )
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert (
            num_patches > MIN_NUM_PATCHES
        ), f"your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size"
        assert pos_embedding_type in ["sine", "none", "learnable", "sine-full"]

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = self.pos_embedding_type == "sine-full"

        self.jacobian_token = jacobian_token
        if jacobian_token:
            ## additional token for jacobian
            num_keypoints = 2 * num_keypoints
            self.num_keypoints = num_keypoints

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h, w = (
            image_size[0] // (4 * self.patch_size[0]),
            image_size[1] // (4 * self.patch_size[1]),
        )
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        # transformer
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            num_keypoints=num_keypoints,
            all_attn=self.all_attn,
        )

        self.to_keypoint_token = nn.Identity()

        if hidden_dim:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.Linear(dim, 2),
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 2),
            )
        trunc_normal_(self.keypoint_token, std=0.02)
        if apply_init:
            self.apply(self._init_weights)

        ## light-weight head for direct jacobian regression
        if estimate_jacobian:
            if hidden_dim:
                self.mlp_head_jacobian = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.Linear(dim, 4),
                )
                self._init_weights(self.mlp_head_jacobian[0])
                self._init_weights(self.mlp_head_jacobian[1])
                self._init_weights(self.mlp_head_jacobian[2])
            else:
                self.mlp_head_jacobian = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, 4),
                )
                self._init_weights(self.mlp_head_jacobian[0])
            self.mlp_head_jacobian[-1].weight.data.zero_()
            self.mlp_head_jacobian[-1].bias.data.copy_(
                torch.tensor([1, 0, 0, 1], dtype=torch.float)
            )
            self.temperature = temperature
            self.affine_jacobian = affine_jacobian
        ## light-weight head for direct keypoint and jacobian regression

    def _make_position_embedding(self, w, h, d_model, pe_type="sine"):
        """
        d_model: embedding size in transformer encoder
        """
        assert pe_type in ["none", "learnable", "sine", "sine-full"]
        if pe_type == "none":
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == "learnable":
                self.pos_embedding = nn.Parameter(
                    torch.zeros(1, self.num_patches + self.num_keypoints, d_model)
                )
                trunc_normal_(self.pos_embedding, std=0.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model), requires_grad=False
                )
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(
        self, d_model, temperature=10000, scale=2 * math.pi
    ):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        # print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img, mask=None):
        p = self.patch_size
        # stem net
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # transformer
        x = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p[0], p2=p[1])
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, "() n d -> b n d", b=b)
        if self.pos_embedding_type in ["sine", "sine-full"]:  #
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        elif self.pos_embedding_type == "learnable":
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, : (n + self.num_keypoints)]
        x = self.dropout(x)

        x = self.transformer(x, mask, self.pos_embedding)
        x_keypoint = self.to_keypoint_token(x[:, 0 : self.num_keypoints // 2])
        keypoint = 2 * F.sigmoid(self.mlp_head(x_keypoint)) - 1
        out = {"value": keypoint}
        if self.mlp_head_jacobian is not None:
            x_jacobian = self.to_keypoint_token(
                x[:, self.num_keypoints // 2 : self.num_keypoints]
            )
            # x_jacobian = x_keypoint
            jacobian = self.mlp_head_jacobian(x_jacobian)
            if self.affine_jacobian:
                theta = jacobian[:, :, 0:2]
                theta = theta / (torch.norm(theta, p=2, dim=-1, keepdim=True) + 1e-10)
                cos_theta = theta[:, :, 0:1]
                sin_theta = theta[:, :, 1:2]
                rotate = torch.cat(
                    (cos_theta, -sin_theta, sin_theta, cos_theta), dim=-1
                )
                rotate = rearrange(rotate, "b c (p1 p2) -> b c p1 p2", p1=2, p2=2)

                scale = torch.abs(jacobian[:, :, 2:]) + 0.1
                # scale = torch.tanh(scale)*0.9+1
                # scale = 1/scale
                scale_x = scale[:, :, 0:1]
                scale_y = scale[:, :, 1:2]
                scale = torch.cat(
                    (
                        scale_x,
                        torch.zeros(scale_x.shape).to(x.device),
                        torch.zeros(scale_x.shape).to(x.device),
                        scale_y,
                    ),
                    dim=-1,
                )
                scale = rearrange(scale, "b c (p1 p2) -> b c p1 p2", p1=2, p2=2)

                jacobian = torch.matmul(rotate, scale)
            else:
                # jacobian = rearrange(jacobian,'b c (p0 p1 p2) -> b c p0 p1 p2',p0=4,p1=self.heatmap_size[0],p2=self.heatmap_size[1])
                # jacobian = heatmap.unsqueeze(2) * jacobian
                # jacobian = jacobian.sum(-1).sum(-1)
                jacobian = rearrange(jacobian, "b c (p1 p2) -> b c p1 p2", p1=2, p2=2)
            out["jacobian"] = jacobian
        return out


class TokenPose_L_base(nn.Module):
    def __init__(
        self,
        *,
        feature_size,
        patch_size,
        num_keypoints,
        dim,
        depth,
        heads,
        mlp_dim,
        apply_init=False,
        hidden_heatmap_dim=64 * 6,
        heatmap_dim=64 * 48,
        heatmap_size=[64, 48],
        channels=3,
        dropout=0.0,
        emb_dropout=0.0,
        pos_embedding_type="learnable",
        estimate_jacobian=False,
        temperature=0.1,
        jacobian_token=True,
        hidden_dim=False,
        affine_jacobian=False,
    ):
        super().__init__()
        assert isinstance(feature_size, list) and isinstance(
            patch_size, list
        ), "image_size and patch_size should be list"
        assert (
            feature_size[0] % patch_size[0] == 0
            and feature_size[1] % patch_size[1] == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (feature_size[0] // (patch_size[0])) * (
            feature_size[1] // (patch_size[1])
        )
        patch_dim = channels * patch_size[0] * patch_size[1]
        assert pos_embedding_type in ["sine", "learnable", "sine-full"]

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = self.pos_embedding_type == "sine-full"

        self.jacobian_token = jacobian_token
        if jacobian_token:
            ## additional token for jacobian
            num_keypoints = 2 * num_keypoints
            self.num_keypoints = num_keypoints
        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h, w = (
            feature_size[0] // (self.patch_size[0]),
            feature_size[1] // (self.patch_size[1]),
        )

        # for normal
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer1 = Transformer(
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            num_keypoints=num_keypoints,
            all_attn=self.all_attn,
            scale_with_head=True,
        )
        self.transformer2 = Transformer(
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            num_keypoints=num_keypoints,
            all_attn=self.all_attn,
            scale_with_head=True,
        )
        self.transformer3 = Transformer(
            dim,
            depth,
            heads,
            mlp_dim,
            dropout,
            num_keypoints=num_keypoints,
            all_attn=self.all_attn,
            scale_with_head=True,
        )

        self.to_keypoint_token = nn.Identity()
        if hidden_dim:
            self.mlp_head = nn.Sequential(
                # nn.LayerNorm(dim),
                # nn.Linear(dim, dim),
                # nn.LayerNorm(dim),
                # nn.Linear(dim, 2),
                nn.LayerNorm(dim * 3),
                nn.Linear(dim * 3, dim * 3),
                nn.LayerNorm(dim * 3),
                nn.Linear(dim * 3, 2),
            )
        else:
            self.mlp_head = nn.Sequential(
                # nn.LayerNorm(dim),
                # nn.Linear(dim, 2),
                nn.LayerNorm(dim * 3),
                nn.Linear(dim * 3, 2),
            )
        trunc_normal_(self.keypoint_token, std=0.02)
        if apply_init:
            self.apply(self._init_weights)

        ## light-weight head for direct jacobian regression
        if estimate_jacobian:
            if hidden_dim:
                self.mlp_head_jacobian = nn.Sequential(
                    # nn.LayerNorm(dim),
                    # nn.Linear(dim, dim),
                    # nn.LayerNorm(dim),
                    # nn.Linear(dim, 4),
                    nn.LayerNorm(dim * 3),
                    nn.Linear(dim * 3, dim * 3),
                    nn.LayerNorm(dim * 3),
                    nn.Linear(dim * 3, 4),
                )
                self._init_weights(self.mlp_head_jacobian[0])
                self._init_weights(self.mlp_head_jacobian[1])
                self._init_weights(self.mlp_head_jacobian[2])
            else:
                self.mlp_head_jacobian = nn.Sequential(
                    # nn.LayerNorm(dim),
                    # nn.Linear(dim, 4),
                    nn.LayerNorm(dim * 3),
                    nn.Linear(dim * 3, 4),
                )
                self._init_weights(self.mlp_head_jacobian[0])
            self.mlp_head_jacobian[-1].weight.data.zero_()
            self.mlp_head_jacobian[-1].bias.data.copy_(
                torch.tensor([1, 0, 0, 1], dtype=torch.float)
            )
            self.temperature = temperature
            self.affine_jacobian = affine_jacobian
        ## light-weight head for direct keypoint and jacobian regression

    def _make_position_embedding(self, w, h, d_model, pe_type="sine"):
        """
        d_model: embedding size in transformer encoder
        """
        assert pe_type in ["none", "learnable", "sine", "sine-full"]
        if pe_type == "none":
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == "learnable":
                self.pos_embedding = nn.Parameter(
                    torch.zeros(1, self.num_patches + self.num_keypoints, d_model)
                )
                trunc_normal_(self.pos_embedding, std=0.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding(d_model), requires_grad=False
                )
                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding(
        self, d_model, temperature=10000, scale=2 * math.pi
    ):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)
        x_embed = area.cumsum(2, dtype=torch.float32)

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_weights(self, m):
        # print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feature, mask=None):
        p = self.patch_size
        # transformer
        x = rearrange(
            feature, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p[0], p2=p[1]
        )
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, "() n d -> b n d", b=b)
        if self.pos_embedding_type in ["sine", "sine-full"]:
            x += self.pos_embedding[:, :n]
            x = torch.cat((keypoint_tokens, x), dim=1)
        else:
            x = torch.cat((keypoint_tokens, x), dim=1)
            x += self.pos_embedding[:, : (n + self.num_keypoints)]
        x = self.dropout(x)

        x1 = self.transformer1(x, mask, self.pos_embedding)
        x2 = self.transformer2(x1, mask, self.pos_embedding)
        x3 = self.transformer3(x2, mask, self.pos_embedding)

        x1_out = self.to_keypoint_token(x1[:, 0 : self.num_keypoints])
        x2_out = self.to_keypoint_token(x2[:, 0 : self.num_keypoints])
        x3_out = self.to_keypoint_token(x3[:, 0 : self.num_keypoints])

        x = torch.cat((x1_out, x2_out, x3_out), dim=2)
        # x = x3

        if self.jacobian_token:
            x_keypoint = self.to_keypoint_token(x[:, 0 : self.num_keypoints // 2])
        else:
            x_keypoint = self.to_keypoint_token(x[:, 0 : self.num_keypoints])
        keypoint = 2 * F.sigmoid(self.mlp_head(x_keypoint)) - 1
        out = {"value": keypoint}
        if self.mlp_head_jacobian is not None:
            if self.jacobian_token:
                x_jacobian = self.to_keypoint_token(
                    x[:, self.num_keypoints // 2 : self.num_keypoints]
                )
            else:
                x_jacobian = x_keypoint
            jacobian = self.mlp_head_jacobian(x_jacobian)
            if self.affine_jacobian:
                theta = jacobian[:, :, 0:2]
                theta = theta / (torch.norm(theta, p=2, dim=-1, keepdim=True) + 1e-10)
                cos_theta = theta[:, :, 0:1]
                sin_theta = theta[:, :, 1:2]
                rotate = torch.cat(
                    (cos_theta, -sin_theta, sin_theta, cos_theta), dim=-1
                )
                rotate = rearrange(rotate, "b c (p1 p2) -> b c p1 p2", p1=2, p2=2)

                scale = torch.abs(jacobian[:, :, 2:]) + 0.1
                # scale = torch.tanh(scale)*0.9+1
                # scale = 1/scale
                scale_x = scale[:, :, 0:1]
                scale_y = scale[:, :, 1:2]
                scale = torch.cat(
                    (
                        scale_x,
                        torch.zeros(scale_x.shape).to(x.device),
                        torch.zeros(scale_x.shape).to(x.device),
                        scale_y,
                    ),
                    dim=-1,
                )
                scale = rearrange(scale, "b c (p1 p2) -> b c p1 p2", p1=2, p2=2)

                jacobian = torch.matmul(rotate, scale)
            else:
                # jacobian = rearrange(jacobian,'b c (p0 p1 p2) -> b c p0 p1 p2',p0=4,p1=self.heatmap_size[0],p2=self.heatmap_size[1])
                # jacobian = heatmap.unsqueeze(2) * jacobian
                # jacobian = jacobian.sum(-1).sum(-1)
                jacobian = rearrange(jacobian, "b c (p1 p2) -> b c p1 p2", p1=2, p2=2)
            out["jacobian"] = jacobian
        return out
