import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

try:
    from NeuFlow.neuflow import NeuFlow
    from NeuFlow.backbone_v7 import ConvBlock
except ImportError:
    NeuFlow = None
    ConvBlock = None
    print("Warning: NeuFlow modules not found. Optical flow extraction will fail.")


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm="BN", activation=nn.LeakyReLU(0.2, inplace=True)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if norm == "BN":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "SN":
            self.conv = nn.utils.spectral_norm(self.conv)
            self.norm = None
        else:
            self.norm = None
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class OpticalFlowEncoder(nn.Module):
    def __init__(self, in_channels=4, flow_cnum=32, norm="BN"):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReplicationPad2d(2),
            ConvBlock2d(in_channels, flow_cnum, kernel_size=5, stride=1, padding=0, norm=norm),
            ConvBlock2d(flow_cnum, flow_cnum * 2, kernel_size=3, stride=2, padding=1, norm=norm),
            ConvBlock2d(flow_cnum * 2, flow_cnum * 2, kernel_size=3, stride=1, padding=1, norm=norm),
            ConvBlock2d(flow_cnum * 2, flow_cnum * 2, kernel_size=3, stride=2, padding=1, norm=norm),
        )
        self.out_channels = flow_cnum * 2

    def forward(self, x):
        return self.net(x)


def fuse_conv_and_bn(conv, bn):
    fused = torch.nn.Conv2d(conv.in_channels, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True).requires_grad_(False).to(conv.weight.device)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused.weight.copy_(torch.mm(w_bn, w_conv).view(fused.weight.shape))
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
    return fused


@torch.no_grad()
def estimate_global_flow_cpu(flow, stride=16, max_iters=2):
    h, w = flow.shape[:2]
    y_indices, x_indices = np.mgrid[0:h:stride, 0:w:stride]
    x_indices = x_indices.flatten()
    y_indices = y_indices.flatten()
    u = flow[y_indices, x_indices, 0]
    v = flow[y_indices, x_indices, 1]
    pts1 = np.stack([x_indices, y_indices], axis=1).astype(np.float32)
    pts2 = pts1 + np.stack([u, v], axis=1).astype(np.float32)
    mask = np.ones(pts1.shape[0], dtype=np.uint8)
    final_transform = None

    for i in range(max_iters):
        if np.sum(mask) < 10:
            break
        thresh = 3.0 - (i * 0.5)
        M, _ = cv2.findHomography(pts1[mask == 1], pts2[mask == 1], cv2.RANSAC, thresh)
        if M is None:
            break
        final_transform = M
        pts1_transformed = cv2.perspectiveTransform(pts1.reshape(-1, 1, 2), M).reshape(-1, 2)
        errors = np.linalg.norm(pts1_transformed - pts2, axis=1)
        median_error = np.median(errors)
        tolerance = 2.5 if i == 0 else 2.0
        new_mask = (errors < (median_error * tolerance + 1e-3)).astype(np.uint8)
        if np.sum(np.abs(new_mask - mask)) / len(mask) < 0.01:
            mask = new_mask
            break
        mask = new_mask

    if final_transform is None:
        return np.zeros_like(flow)

    grid_y, grid_x = np.mgrid[0:h, 0:w]
    points = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(points.reshape(-1, 1, 2), final_transform).reshape(-1, 2)
    global_u = transformed_points[:, 0].reshape(h, w) - grid_x
    global_v = transformed_points[:, 1].reshape(h, w) - grid_y
    return np.stack([global_u, global_v], axis=2).astype(np.float32)


class HomographyBasedEgomotionEstimator(nn.Module):
    def __init__(self, model_path, device="cuda"):
        super().__init__()
        if NeuFlow is None:
            raise ImportError("NeuFlow modules are required for HomographyBasedEgomotionEstimator.")
        self.device = device
        self.flow_model = NeuFlow().to(device)
        checkpoint = torch.load(model_path, map_location="cpu")
        self.flow_model.load_state_dict(checkpoint["model"], strict=True)
        self.flow_model.eval()
        for p in self.flow_model.parameters():
            p.requires_grad = False
        if ConvBlock is not None:
            for m in self.flow_model.modules():
                if type(m) is ConvBlock:
                    m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
                    m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
                    delattr(m, "norm1")
                    delattr(m, "norm2")
                    m.forward = m.forward_fuse
        self.flow_model.half()
        self.current_h, self.current_w, self.current_bs = 384, 384, 1
        self.flow_model.init_bhwd(self.current_bs, self.current_h, self.current_w, device)

    @torch.no_grad()
    def forward(self, frames):
        b, t, c, h, w = frames.shape
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        frames_01 = ((frames + 1.0) * 0.5).half()
        imgs_curr = frames_01[:, :-1].reshape(-1, c, h, w)
        imgs_next = frames_01[:, 1:].reshape(-1, c, h, w)
        if pad_h > 0 or pad_w > 0:
            imgs_curr = F.pad(imgs_curr, (0, pad_w, 0, pad_h))
            imgs_next = F.pad(imgs_next, (0, pad_w, 0, pad_h))
        h_pad, w_pad = imgs_curr.shape[-2:]
        current_bs = imgs_curr.shape[0]
        if (self.current_h != h_pad) or (self.current_w != w_pad) or (self.current_bs != current_bs):
            self.flow_model.init_bhwd(current_bs, h_pad, w_pad, self.device)
            self.current_h, self.current_w, self.current_bs = h_pad, w_pad, current_bs
        flows = self.flow_model(imgs_curr, imgs_next)[-1]
        if pad_h > 0 or pad_w > 0:
            flows = flows[:, :, :h, :w]
        flows_np = flows.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
        g_list = [estimate_global_flow_cpu(flows_np[i], stride=16, max_iters=2) for i in range(flows_np.shape[0])]
        g_np = np.stack(g_list, axis=0)
        l_np = flows_np - g_np
        g = torch.from_numpy(g_np).permute(0, 3, 1, 2).to(self.device)
        l = torch.from_numpy(l_np).permute(0, 3, 1, 2).to(self.device)
        g = g.view(b, t - 1, 2, h, w)
        l = l.view(b, t - 1, 2, h, w)
        return torch.cat([g, g[:, -1:]], dim=1), torch.cat([l, l[:, -1:]], dim=1)


class MGDA(nn.Module):
    def __init__(self, dim, motion_feat_dim, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        in_channels = dim * 2 + motion_feat_dim
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, 2 * kernel_size * kernel_size, kernel_size=1, bias=True),
        )
        self.mask_conv = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, kernel_size * kernel_size, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.dcn = ops.DeformConv2d(dim, dim, kernel_size, padding=self.padding, bias=False)
        nn.init.constant_(self.offset_conv[-1].weight, 0)
        nn.init.constant_(self.offset_conv[-1].bias, 0)

    def get_reference_grid(self, b, h, w, device):
        ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        return torch.stack([xs, ys], dim=0).unsqueeze(0).repeat(b, 1, 1, 1).float()

    def global_warp(self, x, flow):
        b, c, h, w = x.shape
        grid = self.get_reference_grid(b, h, w, x.device)
        vgrid = grid + flow
        vgrid[:, 0] = 2.0 * vgrid[:, 0] / max(w - 1, 1) - 1.0
        vgrid[:, 1] = 2.0 * vgrid[:, 1] / max(h - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        return F.grid_sample(x, vgrid, mode="bilinear", align_corners=True, padding_mode="border")

    def forward(self, patch_feats, global_flow, local_flow, motion_feats):
        coarse = self.global_warp(patch_feats, global_flow)
        concat_feat = torch.cat([coarse, patch_feats, motion_feats], dim=1)
        residual_offset = self.offset_conv(concat_feat)
        modulation_mask = self.mask_conv(concat_feat)
        base_offset = local_flow.repeat(1, self.kernel_size * self.kernel_size, 1, 1)
        total_offset = base_offset + residual_offset
        aligned = self.dcn(coarse, total_offset, modulation_mask)
        return aligned, modulation_mask


class PAMA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.scales = [1, 2, 4]
        self.div_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, len(self.scales), kernel_size=1),
            nn.Softmax(dim=1),
        )
        self.scale_projs = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=3, padding=s, dilation=s, groups=dim) for s in self.scales])
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def compute_divergence(self, flow):
        u = flow[:, 0]
        v = flow[:, 1]
        return (torch.gradient(u, dim=-1)[0] + torch.gradient(v, dim=-2)[0]).unsqueeze(1)

    def forward(self, feat_aligned, global_flow):
        div_map = self.compute_divergence(global_flow)
        scale_weights = self.div_encoder(div_map)
        feat_adaptive = 0
        for i, proj in enumerate(self.scale_projs):
            feat_adaptive = feat_adaptive + proj(feat_aligned) * scale_weights[:, i:i+1]
        q = self.norm_q(feat_aligned.flatten(2).transpose(1, 2))
        kv = self.norm_kv(feat_adaptive.flatten(2).transpose(1, 2))
        attn_out, _ = self.attn(query=q, key=kv, value=kv)
        x = q + attn_out
        x = x + self.ffn(x)
        return x


class EAPA(nn.Module):
    def __init__(self, dim_vid, dim_motion_feat, kernel_size=3, num_heads=8):
        super().__init__()
        self.mgda = MGDA(dim_vid, dim_motion_feat, kernel_size)
        self.pama = PAMA(dim_vid, num_heads)
        self.out_proj = nn.Linear(dim_vid, dim_vid)

    def forward(self, patch_tokens, global_flow, local_flow, motion_feats):
        bt, n, c = patch_tokens.shape
        h = w = int(math.sqrt(n))
        x = patch_tokens.permute(0, 2, 1).reshape(bt, c, h, w)
        if global_flow.shape[-1] != w:
            scale = w / global_flow.shape[-1]
            global_flow = F.interpolate(global_flow, size=(h, w), mode="bilinear", align_corners=False) * scale
            local_flow = F.interpolate(local_flow, size=(h, w), mode="bilinear", align_corners=False) * scale
        x_aligned, align_mask = self.mgda(x, global_flow, local_flow, motion_feats)
        x_enhanced = self.pama(x_aligned, global_flow)
        x_out = self.out_proj(x_enhanced)
        return x_out + patch_tokens, align_mask
