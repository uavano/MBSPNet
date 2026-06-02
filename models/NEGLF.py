import torch
import torch.nn as nn
import torch.nn.functional as F


class BONM(nn.Module):
    def __init__(self, dim, num_memory=200, temp=0.05):
        super().__init__()
        self.dim = dim
        self.num_memory = num_memory
        self.temp = temp
        self.memory = nn.Parameter(torch.randn(num_memory, dim))
        nn.init.orthogonal_(self.memory)

    def forward(self, z, align_mask=None):
        z_norm = F.normalize(z, dim=-1)
        mem_norm = F.normalize(self.memory, dim=-1)
        sim = torch.matmul(z_norm, mem_norm.t())
        weight = F.softmax(sim / self.temp, dim=-1)
        z_mem = torch.matmul(weight, self.memory)

        if align_mask is not None:
            align_mask = align_mask.to(dtype=z.dtype, device=z.device)
            z_out = align_mask * z_mem + (1.0 - align_mask) * z
        else:
            z_out = z_mem

        return z_out, weight

    def compute_ortho_loss(self, global_flow_feat):
        g_norm = F.normalize(global_flow_feat, dim=-1)
        mem_norm = F.normalize(self.memory, dim=-1)
        sim_matrix = torch.matmul(mem_norm, g_norm.t())
        return torch.mean(sim_matrix ** 2)


class SGFA(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm_cls = nn.LayerNorm(dim)
        self.norm_patch = nn.LayerNorm(dim)
        self.aggregation_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.norm_final = nn.LayerNorm(dim)
        self._init_gate_weights()

    def _init_gate_weights(self):
        linear_layer = self.gate[0]
        nn.init.xavier_uniform_(linear_layer.weight, gain=0.01)
        nn.init.constant_(linear_layer.bias, -5.0)

    def forward(self, z_cls, z_patch):
        q = self.norm_cls(z_cls)
        k = self.norm_patch(z_patch)
        v = self.norm_patch(z_patch)
        global_patch_feat, _ = self.aggregation_attn(query=q, key=k, value=v)
        combined = torch.cat([z_cls, global_patch_feat], dim=-1)
        gate_score = self.gate(combined)
        z_refined = z_cls + gate_score * global_patch_feat
        return self.norm_final(z_refined)


class NEGLF(nn.Module):
    def __init__(self, dim, num_memory=200, temp=0.05, num_heads=8, dropout=0.1):
        super().__init__()
        self.bonm = BONM(dim=dim, num_memory=num_memory, temp=temp)
        self.sgfa = SGFA(dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(self, z_cls, z_patch, align_mask=None, global_flow_feat=None, return_aux=False):
        z_patch_refined, memory_weight = self.bonm(z_patch, align_mask=align_mask)

        # SGFA operates at the video/sample level. If patch tokens are provided as
        # [B*T, N, C], reshape them into [B, T*N, C] according to z_cls.
        batch_size = z_cls.size(0)
        if z_patch_refined.size(0) != batch_size:
            z_patch_for_fusion = z_patch_refined.reshape(batch_size, -1, z_patch_refined.size(-1))
        else:
            z_patch_for_fusion = z_patch_refined

        z_fused = self.sgfa(z_cls, z_patch_for_fusion)

        if global_flow_feat is not None:
            ortho_loss = self.bonm.compute_ortho_loss(global_flow_feat)
        else:
            ortho_loss = z_fused.new_tensor(0.0)

        if return_aux:
            return z_fused, {
                "z_patch_refined": z_patch_refined,
                "memory_weight": memory_weight,
                "ortho_loss": ortho_loss,
            }
        return z_fused