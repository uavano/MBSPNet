
import torch
import torch.nn as nn
import torch.nn.functional as F

if __package__:
    from .EAPA import FlowEncoder, DG_GASM
    from .MGTA import TemporalTransformerPatch
    from .NEGLF import A2BO_Memory, SemanticGuidedFusion
else:
    from EAPA import FlowEncoder, DG_GASM
    from MGTA import TemporalTransformerPatch
    from NEGLF import A2BO_Memory, SemanticGuidedFusion


class SpatialPositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(SpatialPositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
    def forward(self, x):
        out = x + self.pos_embedding
        if self.dropout:
            out = self.dropout(out)
        return out


class SpatialMlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(SpatialMlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout2:
            out = self.dropout2(out)
        return out


class SpatialLinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(SpatialLinearGeneral, self).__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))
    def forward(self, x, dims):
        a = torch.tensordot(x, self.weight, dims=dims) + self.bias
        return a


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SpatialSelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5

        self.query = SpatialLinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = SpatialLinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = SpatialLinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = SpatialLinearGeneral((self.heads, self.head_dim), (in_dim,))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None

    def forward(self, x):
        b, n, _ = x.shape

        q = self.query(x, dims=([2], [0]))
        k = self.key(x, dims=([2], [0]))
        v = self.value(x, dims=([2], [0]))

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3)

        out = self.out(out, dims=([2, 3], [0, 1]))

        return out


class SpatialEncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(SpatialEncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SpatialSelfAttention(in_dim, heads=num_heads, dropout_rate=attn_dropout_rate)
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = SpatialMlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out

        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out


class SpatialEncoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(SpatialEncoder, self).__init__()
        self.pos_embedding = SpatialPositionEmbs(num_patches, emb_dim, dropout_rate)
        in_dim = emb_dim
        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = SpatialEncoderBlock(in_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate)
            self.encoder_layers.append(layer)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        out = self.pos_embedding(x)
        for layer in self.encoder_layers:
            out = layer(out)
        out = self.norm(out)
        return out


class SpatialFrameEncoder(nn.Module):
    def __init__(self, image_size, emb_dim, mlp_dim, num_heads, num_layers,
                 dropout_rate=0.1, attn_dropout_rate=0.0):
        super().__init__()
        height, width = image_size
        self.patch_embed = nn.Conv2d(3, emb_dim, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.encoder = SpatialEncoder(
            num_patches=(height // 16) * (width // 16),
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )

    def forward_features(self, x):
        patches = self.patch_embed(x)
        batch, channels, height, width = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1), tokens), dim=1)
        tokens = self.encoder(tokens)
        cls_token = tokens[:, :1, :]
        patches = tokens[:, 1:, :].transpose(1, 2).reshape(
            batch, channels, height, width
        )
        patches = F.avg_pool2d(patches, kernel_size=2, stride=2)
        patches = patches.flatten(2).transpose(1, 2)
        return torch.cat((cls_token, patches), dim=1)

    def forward(self, x):
        return self.forward_features(x)

class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    def forward(self, x):
        return self.dropout(x + self.pos_embedding) if self.dropout else x + self.pos_embedding

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout: out = self.dropout(out)
        out = self.fc2(out)
        if self.dropout: out = self.dropout(out)
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = nn.MultiheadAttention(in_dim, num_heads, dropout=attn_dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)
    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out, _ = self.attn(out, out, out)
        if self.dropout: out = self.dropout(out)
        out += residual
        residual = out
        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out

class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)
        self.encoder_layers = nn.ModuleList([EncoderBlock(emb_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_dim)
    def forward(self, x):
        out = self.pos_embedding(x)
        for layer in self.encoder_layers: out = layer(out)
        out = self.norm(out)
        return out

class Decoder(nn.Module):
    def __init__(self, z_dim=512):
        super(Decoder, self).__init__()
        def Basic(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(intInput, intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(intOutput, intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(intInput, nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(nc, nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(nc, intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )
        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(nc, intOutput, kernel_size=3, stride=2, padding=1, output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        def Upsample2(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(nc, intOutput, kernel_size=4, stride=4, padding=1, output_padding=2),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )
        self.decoder = nn.Sequential(
            Basic(256, 128), Upsample(128, 128), Basic(128, 64), Upsample(64, 64), Upsample2(64, 64), Gen(64, 3, 32),
        )
        self.de_dense = nn.Sequential(nn.Linear(768, 256 * 16 * 16), nn.ELU())
    def forward(self, x):
        if x.dim() == 3: x = x.squeeze(1)
        output = self.de_dense(x)
        output = output.view(output.size(0), 256, 16, 16)
        output = self.decoder(output)
        return output

# =================================================================================
#  Main Model

class MBSPNet(nn.Module):
    def __init__(self, image_size, emb_dim, mlp_dim, num_heads, num_layers, attn_dropout_rate=0.0, dropout_rate=0.1, num_frames=4, nu=0.01):
        super(MBSPNet, self).__init__()
        
        self.spatial_transformer = SpatialFrameEncoder(
            image_size=image_size,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.temporal_transformer_cls = Encoder(num_patches=num_frames, emb_dim=emb_dim, mlp_dim=mlp_dim, num_layers=num_layers, num_heads=num_heads, dropout_rate=dropout_rate, attn_dropout_rate=attn_dropout_rate)
        self.h_grid = image_size[0] // 32
        self.w_grid = image_size[1] // 32
        
        # --- Flow Components ---
        self.flow_encoder = FlowEncoder(in_channels=4, flow_cnum=32, norm='BN')
        self.patch2vec = nn.Conv2d(64, emb_dim, kernel_size=8, stride=8, padding=0)
        self.dg_gasm = DG_GASM(dim_vid=emb_dim, dim_motion_feat=emb_dim, kernel_size=3, num_heads=8)

        self.global_ortho_enc = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),            
            nn.Linear(32, emb_dim)   
        )

        self.memory_module = A2BO_Memory(dim=emb_dim, num_memory=200)

        self.temporal_transformer_patch = TemporalTransformerPatch(
            token_size=(self.h_grid, self.w_grid), frame_hidden=emb_dim, num_heads=8, t_groupSize=num_frames, mlp_ratio=4, dropout=dropout_rate, n_vecs=1, 
            t2t_params={'kernel_size': [3,3], 'stride': 1, 'padding': 1, 'output_size': (self.h_grid, self.w_grid)}
        )

        self.fusion_module = SemanticGuidedFusion(dim=emb_dim, num_heads=8, dropout=dropout_rate)
        self.decoder = Decoder()
        
        self.nu = nu
        self.radius = torch.nn.Parameter(torch.tensor([0.0]))
        self.center = torch.nn.Parameter(torch.randn(emb_dim))

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
        feat_cls_residual = self.cross_att(feat_cls_enc)
        z_cls_final = feat_cls_enc[:, 0:1, :] + feat_cls_residual
        
        flow_tokens_flat = None
        align_mask = None
        ortho_loss = torch.tensor(0.0).to(x.device)
        
        if global_flows is not None and local_flows is not None:
            g_flow = global_flows.reshape(b * t, 2, h_img, w_img)
            l_flow = local_flows.reshape(b * t, 2, h_img, w_img)
            
            joint_flow = torch.cat([g_flow, l_flow], dim=1) 
            motion_feat = self.flow_encoder(joint_flow)
            motion_token_map = self.patch2vec(motion_feat)
            
            z_spatial_enhanced, align_mask_spatial = self.dg_gasm(spatial_patch_tokens, g_flow, l_flow, motion_token_map)
            
            align_mask = align_mask_spatial.mean(dim=1, keepdim=True).flatten(2).transpose(1, 2)
            
            flow_tokens_flat = motion_token_map.flatten(2).transpose(1, 2)
            
            g_flow_embed = self.global_ortho_enc(g_flow) 
            ortho_loss = self.memory_module.compute_ortho_loss(g_flow_embed)
            
        else:
            z_spatial_enhanced = spatial_patch_tokens
            flow_tokens_flat = torch.zeros_like(spatial_patch_tokens)
            align_mask = torch.ones_like(spatial_patch_tokens[..., :1])

        z_patch_seq = self.temporal_transformer_patch(
            z_spatial_enhanced, 
            flow_tokens_flat, 
            t, self.h_grid, self.w_grid, (self.h_grid, self.w_grid)
        )

        z_patch_mem, att_weight = self.memory_module(z_patch_seq, align_mask)
        z_patch_flat = z_patch_mem.view(b, t * (self.h_grid * self.w_grid), -1) 
        z_final = self.fusion_module(z_cls_final, z_patch_flat)
        output = self.decoder(z_final)
        
        return output, ortho_loss