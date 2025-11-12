"""
Implements the TransFuser vision backbone.
"""

import copy
import math
from typing import List, Tuple

import timm
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoProcessor, AutoModel

from navsim.agents.transfuser.transfuser_config import TransfuserConfig


def build_2d_sincos_pos_embed(
    height: int, width: int, dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Utility to create 2D sine/cosine positional embeddings (DETR style)."""

    if dim % 4 != 0:
        raise ValueError("dim must be divisible by 4 for 2D sine/cosine embeddings.")

    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )

    omega = torch.arange(dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / (dim // 4)))

    pos_y = torch.einsum("hw,c->hwc", y_grid, omega)
    pos_x = torch.einsum("hw,c->hwc", x_grid, omega)

    pos = torch.cat(
        [
            torch.sin(pos_y),
            torch.cos(pos_y),
            torch.sin(pos_x),
            torch.cos(pos_x),
        ],
        dim=-1,
    )

    return pos.view(-1, dim)


class TransfuserBackbone(nn.Module):
    """Multi-scale Fusion Transformer for image + LiDAR feature fusion."""

    def __init__(self, config: TransfuserConfig):

        super().__init__()
        self.config = config

        if self.config.backbone == "resnet":
            self.image_encoder = timm.create_model(
                config.image_architecture, pretrained=True, features_only=True
            )
            if config.use_ground_plane:
                in_channels = 2 * config.lidar_seq_len
            else:
                in_channels = config.lidar_seq_len

            if config.latent:
                self.lidar_latent = nn.Parameter(
                    torch.randn(
                        (
                            1,
                            in_channels,
                            config.lidar_resolution_width,
                            config.lidar_resolution_height,
                        ),
                        requires_grad=True,
                    )
                )

            self.avgpool_img = nn.AdaptiveAvgPool2d(
                (self.config.img_vert_anchors, self.config.img_horz_anchors)
            )

            self.lidar_encoder = timm.create_model(
                config.lidar_architecture,
                pretrained=False,
                in_chans=in_channels,
                features_only=True,
            )
            self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
            self.avgpool_lidar = nn.AdaptiveAvgPool2d(
                (self.config.lidar_vert_anchors, self.config.lidar_horz_anchors)
            )
            lidar_time_frames = [1, 1, 1, 1]

            self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
            start_index = 0
            # Some networks have a stem layer
            if len(self.image_encoder.return_layers) > 4:
                start_index += 1

            self.transformers = nn.ModuleList(
                [
                    GPT(
                        n_embd=self.image_encoder.feature_info.info[start_index + i][
                            "num_chs"
                        ],
                        config=config,
                        # lidar_video=self.lidar_video,
                        lidar_time_frames=lidar_time_frames[i],
                    )
                    for i in range(4)
                ]
            )
            self.lidar_channel_to_img = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.lidar_encoder.feature_info.info[start_index + i][
                            "num_chs"
                        ],
                        self.image_encoder.feature_info.info[start_index + i][
                            "num_chs"
                        ],
                        kernel_size=1,
                    )
                    for i in range(4)
                ]
            )
            self.img_channel_to_lidar = nn.ModuleList(
                [
                    nn.Conv2d(
                        self.image_encoder.feature_info.info[start_index + i][
                            "num_chs"
                        ],
                        self.lidar_encoder.feature_info.info[start_index + i][
                            "num_chs"
                        ],
                        kernel_size=1,
                    )
                    for i in range(4)
                ]
            )

            self.num_image_features = self.image_encoder.feature_info.info[
                start_index + 3
            ]["num_chs"]
            # Typical encoders down-sample by a factor of 32
            self.perspective_upsample_factor = (
                self.image_encoder.feature_info.info[start_index + 3]["reduction"]
                // self.config.perspective_downsample_factor
            )

            if self.config.transformer_decoder_join:
                self.num_features = self.lidar_encoder.feature_info.info[
                    start_index + 3
                ]["num_chs"]
            else:
                if self.config.add_features:
                    self.lidar_to_img_features_end = nn.Linear(
                        self.lidar_encoder.feature_info.info[start_index + 3][
                            "num_chs"
                        ],
                        self.image_encoder.feature_info.info[start_index + 3][
                            "num_chs"
                        ],
                    )
                    # Number of features the encoder produces.
                    self.num_features = self.image_encoder.feature_info.info[
                        start_index + 3
                    ]["num_chs"]
                else:
                    # Number of features the encoder produces.
                    self.num_features = (
                        self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
                        + self.lidar_encoder.feature_info.info[start_index + 3][
                            "num_chs"
                        ]
                    )

            # FPN fusion
            channel = self.config.bev_features_channels
            self.relu = nn.ReLU(inplace=True)
            # top down
            if self.config.detect_boxes or self.config.use_bev_semantic:
                self.upsample = nn.Upsample(
                    scale_factor=self.config.bev_upsample_factor,
                    mode="bilinear",
                    align_corners=False,
                )
                self.upsample2 = nn.Upsample(
                    size=(
                        self.config.lidar_resolution_height
                        // self.config.bev_down_sample_factor,
                        self.config.lidar_resolution_width
                        // self.config.bev_down_sample_factor,
                    ),
                    mode="bilinear",
                    align_corners=False,
                )

                self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
                self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)

                # lateral
                self.c5_conv = nn.Conv2d(
                    self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"],
                    channel,
                    (1, 1),
                )
        elif self.config.backbone == "ijepa":
            self.lidar_encoder = timm.create_model(
                config.lidar_architecture,
                pretrained=False,
                in_chans=config.lidar_seq_len,
                features_only=True,
            )

            lidar_stage_channels = self.lidar_encoder.feature_info.channels()
            if len(lidar_stage_channels) < 4:
                raise ValueError(
                    "Expected LiDAR encoder to expose at least 4 stages for fusion."
                )
            lidar_stage_channels = lidar_stage_channels[-4:]

            self.image_encoder = IJEPABackbone(
                self.config, stage_dims=lidar_stage_channels
            )

            fusion_dim = getattr(self.config, "ijepa_fusion_dim", 256)
            fusion_heads = getattr(self.config, "ijepa_fusion_heads", 8)
            fusion_latents = getattr(self.config, "ijepa_fusion_latents", 64)
            fusion_layers = getattr(self.config, "ijepa_fusion_layers", 2)

            self.ms_fusion = nn.ModuleList(
                [
                    CrossAttentionFusion(
                        vision_dim_raw=lidar_stage_channels[idx],
                        fusion_dim=fusion_dim,
                        lidar_feature_dim=lidar_stage_channels[idx],
                        output_dim=lidar_stage_channels[idx],
                        num_latents=fusion_latents,
                        num_layers=fusion_layers,
                        num_heads=fusion_heads,
                    )
                    for idx in range(4)
                ]
            )

            channel = self.config.bev_features_channels
            self.lidar_to_bev = nn.Conv2d(
                lidar_stage_channels[-1], channel, kernel_size=1
            )
            self.num_features = channel

            self.relu = nn.ReLU(inplace=True)

            self.upsample = nn.Upsample(
                scale_factor=self.config.bev_upsample_factor,
                mode="bilinear",
                align_corners=False,
            )
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_resolution_height
                    // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width
                    // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )

            self.up_conv5 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
            self.up_conv4 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

            self.pool_to_8x8 = nn.AdaptiveAvgPool2d((8, 8))

            self.det_head = nn.Conv2d(
                channel,
                self.config.num_bounding_boxes,
                kernel_size=1,
            )
        else:
            raise NotImplementedError(
                f"Backbone {self.config.backbone} not implemented"
            )

    def top_down(self, x):

        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))

        return p3

    def forward(self, image, lidar):
        """
        Returns:
            features            -> high-res BEV map for heads (B, C, H_supervise, W_supervise)
            fused_features      -> global pooled embedding vector (B, C*8*8)
            image_feature_grid  -> camera spatial features for visualization/losses
        """
        if self.config.backbone == "ijepa":
            # ----- 1. Vision branch (4-stage pyramid) -----
            image_feats, image_feature_grid = self.image_encoder(
                image
            )  # pyramid: List[Tensor], feat: Tensor

            # ----- 2. LiDAR branch (4-stage pyramid) -----
            lidar_feats_all = self.lidar_encoder(lidar)
            if len(lidar_feats_all) < 4:
                raise ValueError(
                    "LiDAR encoder did not return enough feature maps for fusion."
                )
            lidar_feats = list(lidar_feats_all[-4:])

            # ----- 3. Multi-scale fusion (4 stages) -----
            for idx, fusion_block in enumerate(self.ms_fusion):
                img_stage = image_feats[idx]
                lidar_stage = lidar_feats[idx]

                # Validate LiDAR stage dimensions (typical ResNet stages)
                assert lidar_stage.shape[-2:] in {
                    (64, 64),
                    (32, 32),
                    (16, 16),
                    (8, 8),
                }, f"Stage {idx}: unexpected LiDAR shape {lidar_stage.shape[-2:]}"

                # Ensure same device
                if img_stage.device != lidar_stage.device:
                    img_stage = img_stage.to(lidar_stage.device, non_blocking=True)

                # Ensure spatial dims match
                if img_stage.shape[-2:] != lidar_stage.shape[-2:]:
                    img_stage = F.interpolate(
                        img_stage,
                        size=lidar_stage.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # Fuse and update LiDAR features
                lidar_feats[idx] = fusion_block(
                    image_features=img_stage, lidar_features=lidar_stage
                )

            # ----- 4. Project top stage to BEV channels -----
            top_stage = lidar_feats[-1]
            bev_feature = self.lidar_to_bev(top_stage)

            # ----- 5. Upsample / refinement for detection/seg heads -----
            up = self.upsample(bev_feature)
            up = self.relu(up)
            up = self.up_conv5(up)

            up2 = self.upsample2(up)
            up2 = self.relu(up2)
            up2 = self.up_conv4(up2)

            bev_feature_upscale = up2  # high-res BEV (supervision head space)

            # ----- 6. Global pooled embedding -----
            pooled_8x8 = self.pool_to_8x8(bev_feature)  # (B, C, 8, 8)
            fused_features = torch.flatten(pooled_8x8, 1)

            # match resnet return signature
            return bev_feature_upscale, bev_feature, image_feature_grid

        elif self.config.backbone == "resnet":
            # Original ResNet multi-scale fusion with GPT blocks
            image_features, lidar_features = image, lidar

            if self.config.latent and lidar_features is None:
                batch_size = image.shape[0]
                lidar_features = self.lidar_latent.repeat(batch_size, 1, 1, 1)

            image_layers = iter(self.image_encoder.items())
            lidar_layers = iter(self.lidar_encoder.items())

            if len(self.image_encoder.return_layers) > 4:
                image_features = self.forward_layer_block(
                    image_layers, self.image_encoder.return_layers, image_features
                )
            if len(self.lidar_encoder.return_layers) > 4:
                lidar_features = self.forward_layer_block(
                    lidar_layers, self.lidar_encoder.return_layers, lidar_features
                )

            for i in range(4):
                image_features = self.forward_layer_block(
                    image_layers, self.image_encoder.return_layers, image_features
                )
                lidar_features = self.forward_layer_block(
                    lidar_layers, self.lidar_encoder.return_layers, lidar_features
                )
                image_features, lidar_features = self.fuse_features(
                    image_features, lidar_features, i
                )

            x4 = lidar_features
            image_feature_grid = image_features

            if self.config.transformer_decoder_join:
                fused_features = lidar_features
            else:
                image_features = self.global_pool_img(image_features)
                lidar_features = self.global_pool_lidar(lidar_features)

                if self.config.add_features:
                    lidar_features_layer = self.lidar_to_img_features_end(
                        lidar_features.flatten(1)
                    )
                    fused_features = torch.cat(
                        [image_features.flatten(1), lidar_features_layer], dim=1
                    )
                else:
                    fused_features = torch.cat(
                        [image_features.flatten(1), lidar_features.flatten(1)], dim=1
                    )

            if self.config.detect_boxes or self.config.use_bev_semantic:
                features = self.top_down(x4)
            else:
                features = None

            return features, fused_features, image_feature_grid

    def fuse_features(self, image_features, lidar_features, i):
        """
        Perform a TransFuser feature fusion block using a Transformer module.
        :param image_features: Features from the image branch
        :param lidar_features: Features from the LiDAR branch
        :param i: Transformer layer index.
        :return: image_features and lidar_features with added features from the other branch.
        """
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)

        lidar_embd_layer = self.lidar_channel_to_img[i](lidar_embd_layer)

        image_features_layer, lidar_features_layer = self.transformers[i](
            image_embd_layer, lidar_embd_layer
        )
        lidar_features_layer = self.img_channel_to_lidar[i](lidar_features_layer)

        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features


class GPT(nn.Module):
    """The full GPT language backbone, with a context size of block_size."""

    # def __init__(self, n_embd, config, lidar_video, lidar_time_frames):
    def __init__(self, n_embd, config, lidar_time_frames):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1
        self.lidar_seq_len = config.lidar_seq_len
        self.config = config
        self.lidar_time_frames = lidar_time_frames

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(
            torch.zeros(
                1,
                self.seq_len
                * self.config.img_vert_anchors
                * self.config.img_horz_anchors
                + lidar_time_frames
                * self.config.lidar_vert_anchors
                * self.config.lidar_horz_anchors,
                self.n_embd,
            )
        )

        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embd,
                    config.n_head,
                    config.block_exp,
                    config.attn_pdrop,
                    config.resid_pdrop,
                )
                for layer in range(config.n_layer)
            ]
        )

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
        """

        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]

        img_h, img_w = image_tensor.shape[2:4]

        assert self.seq_len == 1
        image_tensor = (
            image_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )
        lidar_tensor = (
            lidar_tensor.permute(0, 2, 3, 1).contiguous().view(bz, -1, self.n_embd)
        )

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        image_tensor_out = (
            x[
                :,
                : self.seq_len
                * self.config.img_vert_anchors
                * self.config.img_horz_anchors,
                :,
            ]
            .view(bz * self.seq_len, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        lidar_tensor_out = (
            x[
                :,
                self.seq_len
                * self.config.img_vert_anchors
                * self.config.img_horz_anchors :,
                :,
            ]
            .view(bz, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the
    end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        b, t, c = x.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        k = (
            self.key(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        q = (
            self.query(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        v = (
            self.value(x).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = (
            y.transpose(1, 2).contiguous().view(b, t, c)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x


class MultiheadAttentionWithAttention(nn.Module):
    """
    MultiheadAttention that also return attention weights
    """

    def __init__(self, n_embd, n_head, pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(pdrop)
        self.resid_drop = nn.Dropout(pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, q_in, k_in, v_in):
        b, t, c = q_in.size()
        _, t_mem, _ = k_in.size()

        # calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim
        q = (
            self.query(q_in).view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        k = (
            self.key(k_in).view(b, t_mem, self.n_head, c // self.n_head).transpose(1, 2)
        )  # (b, nh, t, hs)
        v = (
            self.value(v_in)
            .view(b, t_mem, self.n_head, c // self.n_head)
            .transpose(1, 2)
        )  # (b, nh, t, hs)

        # self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (b, nh, t, t) x (b, nh, t, hs) -> (b, nh, t, hs)
        y = (
            y.transpose(1, 2).contiguous().view(b, t, c)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        attention = torch.mean(att, dim=1)  # Average attention over heads
        return y, attention


class TransformerDecoderLayerWithAttention(nn.Module):
    """A Transformer decoder that returns the attentions."""

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
    ):
        super().__init__()
        self.self_attn = MultiheadAttentionWithAttention(d_model, nhead, dropout)
        self.multihead_attn = MultiheadAttentionWithAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, tgt, memory):
        x = tgt
        tmp, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(tmp))
        tmp, attention = self.multihead_attn(x, memory, memory)
        x = self.norm2(x + self.dropout2(tmp))
        tmp = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(tmp))

        return x, attention


class TransformerDecoderWithAttention(nn.Module):
    """A Transformer decoder that returns the attentions."""

    def __init__(self, layers, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layers) for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, queries, memory):
        output = queries
        attentions = []
        for mod in self.layers:
            output, attention = mod(output, memory)
            attentions.append(attention)

        if self.norm is not None:
            output = self.norm(output)

        avg_attention = torch.mean(torch.stack(attentions), dim=0)
        return output, avg_attention


class IJEPABackbone(nn.Module):
    def __init__(self, config: TransfuserConfig, stage_dims: List[int]):
        super().__init__()

        self.config = config
        self.stage_dims = stage_dims

        self.processor = AutoProcessor.from_pretrained(
            self.config.ijepa_model_id, use_fast=True
        )

        img_mean = getattr(self.processor, "image_mean", [0.5, 0.5, 0.5])
        img_std = getattr(self.processor, "image_std", [0.5, 0.5, 0.5])

        self.register_buffer(
            "img_mean", torch.tensor(img_mean, dtype=torch.float32).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "img_std", torch.tensor(img_std, dtype=torch.float32).view(1, -1, 1, 1)
        )

        self.vision_tower = AutoModel.from_pretrained(
            self.config.ijepa_model_id, output_hidden_states=True
        )
        self.vision_tower.eval()
        for param in self.vision_tower.parameters():
            param.requires_grad = False

        self.embed_dim = getattr(
            self.vision_tower.config,
            "hidden_size",
            getattr(self.vision_tower.config, "embed_dim", 1280),
        )

        self.patch_size = None
        if hasattr(self.vision_tower.config, "patch_size"):
            ps = self.vision_tower.config.patch_size
            self.patch_size = ps if isinstance(ps, int) else ps[0]

        self.stage_projs = nn.ModuleList(
            [nn.Conv2d(self.embed_dim, dim, kernel_size=1) for dim in stage_dims]
        )
        self.stage_norms = nn.ModuleList([nn.GroupNorm(8, dim) for dim in stage_dims])
        self.stage_pools = nn.ModuleList(
            [
                nn.AvgPool2d(kernel_size=2, stride=2)
                for _ in range(max(len(stage_dims) - 1, 0))
            ]
        )

    def forward(self, x_bchw: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = x_bchw
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(
                x.float() if x.dtype == torch.uint8 else x,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        img_mean = self.img_mean.to(x.device, non_blocking=True)
        img_std = self.img_std.to(x.device, non_blocking=True)
        x = (x - img_mean) / img_std

        vt_dev = next(self.vision_tower.parameters()).device
        if vt_dev != x.device:
            self.vision_tower.to(x.device, non_blocking=True)

        # Frozen encoder - no gradients needed for memory efficiency
        with torch.no_grad():
            outputs = self.vision_tower(pixel_values=x)
        tokens = outputs.last_hidden_state

        batch, num_tokens, _ = tokens.shape
        height, width = x.shape[-2:]

        patch = getattr(self, "patch_size", None)
        if patch is None:
            tokens_without_cls = num_tokens - 1
            guess = (
                int(round(((height * width) / tokens_without_cls) ** 0.5))
                if tokens_without_cls > 0
                else None
            )
            if guess and height % guess == 0 and width % guess == 0:
                tokens_core = tokens[:, 1:, :]
                patch = guess
            else:
                guess = int(round(((height * width) / num_tokens) ** 0.5))
                if height % guess != 0 or width % guess != 0:
                    raise RuntimeError(
                        f"Unable to infer patch size for I-JEPA output (H={height}, W={width}, N={num_tokens})."
                    )
                tokens_core = tokens
                patch = guess
            self.patch_size = patch
        else:
            expected_tokens = (height // patch) * (width // patch)
            if num_tokens == expected_tokens + 1:
                tokens_core = tokens[:, 1:, :]
            elif num_tokens == expected_tokens:
                tokens_core = tokens
            else:
                raise RuntimeError(
                    f"Token count {num_tokens} incompatible with patch size {patch} (expected {expected_tokens} or {expected_tokens + 1})."
                )

        grid_h = height // self.patch_size
        grid_w = width // self.patch_size
        feat = tokens_core.transpose(1, 2).reshape(
            batch, self.embed_dim, grid_h, grid_w
        )
        pyramid: List[torch.Tensor] = []
        current = feat
        for idx, proj in enumerate(self.stage_projs):
            projected = proj(current)
            projected = self.stage_norms[idx](
                projected
            )  # Apply GroupNorm for stability
            pyramid.append(projected)
            if idx < len(self.stage_pools):
                current = self.stage_pools[idx](current)

        return pyramid, feat


class CrossAttentionFusion(nn.Module):
    """
    Perceiver-style cross-attention to fuse vision tokens (I-JEPA) with LiDAR BEV features.

    Args:
        vision_dim_raw: channels coming out of IJEPABackbone (e.g. 1280 for ViT-H/14).
        fusion_dim: internal dim we run attention in (e.g. 256). Also dim of latent_queries.
        lidar_feature_dim: channels in lidar_features (B, C_l, H, W) fed here.
        output_dim: channels you want to hand back to downstream BEV head.
        num_latents: number of learned latent queries.
        num_layers: number of cross-attention refinement layers.
    """

    def __init__(
        self,
        vision_dim_raw: int,
        fusion_dim: int,
        lidar_feature_dim: int,
        output_dim: int,
        num_latents: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()

        self.num_latents = num_latents
        self.output_dim = output_dim
        self.fusion_dim = fusion_dim
        self.lidar_feature_dim = lidar_feature_dim

        # project IJEPA patch embeddings -> fusion_dim (so MHA dims line up)
        self.image_proj = nn.Linear(vision_dim_raw, fusion_dim)

        # learnable latent array: (num_latents, fusion_dim)
        self.latent_queries = nn.Parameter(torch.randn(num_latents, fusion_dim))

        # Pre-LayerNorm for training stability
        self.pre_ln = nn.LayerNorm(fusion_dim)

        # stack of cross-attn blocks, batch_first=True gives (B, L, C)
        self.image_cross_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=fusion_dim,
                    num_heads=num_heads,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # final projection that mixes attended img signal with lidar BEV grid
        # we concatenate along channel dim after broadcasting the attended img vec
        self.fusion_projection = nn.Linear(fusion_dim + lidar_feature_dim, output_dim)

    def forward(self, image_features, lidar_features):
        """
        image_features: either (B, N, C_raw) tokens OR (B, C_raw, H_img, W_img) fmap
        lidar_features: (B, C_l, H, W) BEV / lidar feature map
        returns:
            fused_bev: (B, output_dim, H, W)
        """
        dev = self.latent_queries.device

        # Ensure tensors are on same device
        image_features = image_features.to(dev, non_blocking=True)
        lidar_features = lidar_features.to(dev, non_blocking=True)

        # 1. Flatten vision features to token sequence if still 4D
        if image_features.dim() == 4:
            B, C_raw, H_img, W_img = image_features.shape
            image_features = image_features.permute(0, 2, 3, 1).reshape(
                B, H_img * W_img, C_raw
            )
        else:
            B, N, C_raw = image_features.shape
            spatial_dim = int(round(N**0.5))
            if spatial_dim * spatial_dim != N:
                raise ValueError(
                    f"Cannot infer square spatial shape from token count {N}."
                )
            H_img, W_img = spatial_dim, spatial_dim

        # 2. Project vision tokens to fusion_dim
        image_features = self.image_proj(image_features)  # (B, N, fusion_dim)

        pos_image = build_2d_sincos_pos_embed(
            H_img,
            W_img,
            self.fusion_dim,
            device=image_features.device,
            dtype=image_features.dtype,
        )
        image_features = image_features + pos_image.unsqueeze(0)

        # 3. Build latent queries for this batch
        # latent_queries: (num_latents, fusion_dim) -> (B, num_latents, fusion_dim)
        latents = self.latent_queries.unsqueeze(0).expand(B, -1, -1).contiguous()

        # 4. Cross-attention refinement with pre-LayerNorm
        # After loop: latents is (B, num_latents, fusion_dim)
        for attn in self.image_cross_attention:
            # Apply pre-LN for training stability
            latents_normed = self.pre_ln(latents)
            # query = latents (what we want to learn)
            # key/value = image_features (what we observe from camera encoder)
            attended, _ = attn(
                query=latents_normed,
                key=image_features,
                value=image_features,
            )
            # Residual connection
            latents = latents + attended

        # 5. Pool latents down to a single global camera vector for this timestep
        img_vec = latents.mean(dim=1)  # (B, fusion_dim)

        # 6. Broadcast camera vector over LiDAR BEV grid
        B_l, _, H, W = lidar_features.shape
        assert B_l == B, "batch mismatch vision vs lidar"

        pos_lidar = build_2d_sincos_pos_embed(
            H,
            W,
            self.lidar_feature_dim,
            device=lidar_features.device,
            dtype=lidar_features.dtype,
        )
        pos_lidar = pos_lidar.view(1, H, W, self.lidar_feature_dim).permute(0, 3, 1, 2)
        lidar_features = lidar_features + pos_lidar

        img_grid = (
            img_vec.unsqueeze(-1).unsqueeze(-1).expand(B, self.fusion_dim, H, W)
        )  # (B, fusion_dim, H, W)

        # 7. Channel concat and linear fusion head
        fused = torch.cat(
            [img_grid, lidar_features], dim=1
        )  # (B, fusion_dim + C_l, H, W)

        # fusion_projection is Linear(...), which expects last dim = in_features.
        # We'll apply it per-pixel, so reshape -> project -> reshape back.
        fused = fused.permute(0, 2, 3, 1)  # (B, H, W, fusion_dim + C_l)
        fused = self.fusion_projection(fused)  # (B, H, W, output_dim)
        fused = fused.permute(0, 3, 1, 2).contiguous()  # (B, output_dim, H, W)

        return fused
