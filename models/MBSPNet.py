import torch
import torch.nn as nn
import timm

from models.common import Encoder
from models.EAPA import OpticalFlowEncoder, EAPA
from models.NEGLF import NEGLF
from models.TCTA import TemporalTransformerPatch


class Decoder(nn.Module):
    def __init__(self, z_dim=768):
        super().__init__()

        def basic(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=False),
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=False),
            )

        def gen(in_c, out_c, nc):
            return nn.Sequential(
                nn.Conv2d(in_c, nc, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(nc),
                nn.ReLU(inplace=False),
                nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(nc),
                nn.ReLU(inplace=False),
                nn.Conv2d(nc, out_c, kernel_size=3, stride=1, padding=1),
                nn.Tanh(),
            )

        def upsample(nc, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(nc, out_c, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=False),
            )

        def upsample2(nc, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(nc, out_c, kernel_size=4, stride=4, padding=1, output_padding=2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=False),
            )

        self.decoder = nn.Sequential(basic(256, 128), upsample(128, 128), basic(128, 64), upsample(64, 64), upsample2(64, 64), gen(64, 3, 32))
        self.de_dense = nn.Sequential(nn.Linear(z_dim, 256 * 16 * 16), nn.ELU())

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        x = self.de_dense(x)
        x = x.view(x.size(0), 256, 16, 16)
        return self.decoder(x)


class MBSPNet(nn.Module):
    def __init__(self, image_size=(384, 384), emb_dim=768, mlp_dim=3072, num_heads=12, num_layers=12, dropout_rate=0.1, attn_dropout_rate=0.0, num_frames=4, nu=0.01, memory_size=200):
        super().__init__()
        self.spatial_transformer = timm.create_model("vit_base_patch32_384", num_classes=0, pretrained=False, img_size=image_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.temporal_transformer_cls = Encoder(num_patches=num_frames, emb_dim=emb_dim, mlp_dim=mlp_dim, num_layers=num_layers, num_heads=num_heads, dropout_rate=dropout_rate, attn_dropout_rate=attn_dropout_rate)
        self.h_grid = image_size[0] // 32
        self.w_grid = image_size[1] // 32
        self.flow_encoder = OpticalFlowEncoder(in_channels=4, flow_cnum=32, norm="BN")
        self.patch2vec = nn.Conv2d(64, emb_dim, kernel_size=8, stride=8, padding=0)
        self.eapa = EAPA(dim_vid=emb_dim, dim_motion_feat=emb_dim, kernel_size=3, num_heads=8)
        self.global_ortho_enc = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, emb_dim),
        )
        self.tcta = TemporalTransformerPatch(token_size=(self.h_grid, self.w_grid), frame_hidden=emb_dim, num_heads=8, t_group_size=num_frames, mlp_ratio=4, dropout=dropout_rate, n_vecs=1, t2t_params={"kernel_size": [3, 3], "stride": 1, "padding": 1, "output_size": (self.h_grid, self.w_grid)})
        self.neglf = NEGLF(dim=emb_dim, num_memory=memory_size, num_heads=8, dropout=dropout_rate)
        self.decoder = Decoder(z_dim=emb_dim)
        self.nu = nu
        self.radius = nn.Parameter(torch.tensor([0.0]))
        self.center = nn.Parameter(torch.randn(emb_dim))

    def forward(self, x, global_flows=None, local_flows=None):
        x = x.float()
        b, t, c, h_img, w_img = x.shape
        x_flat = x.reshape(b * t, c, h_img, w_img)
        spatial_features = self.spatial_transformer.forward_features(x_flat)
        spatial_cls_tokens = spatial_features[:, 0:1, :]
        spatial_patch_tokens = spatial_features[:, 1:, :]
        cls_seq = spatial_cls_tokens.view(b, t, -1)
        temporal_cls_token = self.cls_token.repeat(b, 1, 1)
        emb_cls = torch.cat([temporal_cls_token, cls_seq], dim=1)
        feat_cls_enc = self.temporal_transformer_cls(emb_cls)
        z_cls_final = feat_cls_enc[:, 0:1, :] + feat_cls_enc

        align_mask = None
        flow_tokens_flat = None
        g_flow_embed = None
        if global_flows is not None and local_flows is not None:
            g_flow = global_flows.reshape(b * t, 2, h_img, w_img)
            l_flow = local_flows.reshape(b * t, 2, h_img, w_img)
            motion_feat = self.flow_encoder(torch.cat([g_flow, l_flow], dim=1))
            motion_token_map = self.patch2vec(motion_feat)
            z_spatial_enhanced, align_mask_spatial = self.eapa(spatial_patch_tokens, g_flow, l_flow, motion_token_map)
            align_mask = align_mask_spatial.mean(dim=1, keepdim=True).flatten(2).transpose(1, 2)
            flow_tokens_flat = motion_token_map.flatten(2).transpose(1, 2)
            g_flow_embed = self.global_ortho_enc(g_flow)
        else:
            z_spatial_enhanced = spatial_patch_tokens
            flow_tokens_flat = torch.zeros_like(spatial_patch_tokens)
            align_mask = torch.ones_like(spatial_patch_tokens[..., :1])

        z_patch_seq = self.tcta(z_spatial_enhanced, flow_tokens_flat, t, self.h_grid, self.w_grid, (self.h_grid, self.w_grid))
        z_final, aux = self.neglf(z_cls_final, z_patch_seq, align_mask=align_mask, global_flow_feat=g_flow_embed, return_aux=True)
        ortho_loss = aux["ortho_loss"]
        output = self.decoder(z_final)
        return output, ortho_loss
