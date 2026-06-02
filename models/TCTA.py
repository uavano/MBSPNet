import math
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCTA(nn.Module):
    def __init__(self, token_size, group_size, d_model, head, p=0.1):
        super().__init__()
        self.h, self.w = token_size
        self.group_size = group_size
        self.d_model = d_model
        self.head = head
        self.p = p
        self.wh, self.ww = math.ceil(self.h / self.group_size), math.ceil(self.w / self.group_size)
        self.pad_r = (self.ww * self.group_size) - self.w
        self.pad_b = (self.wh * self.group_size) - self.h
        self.new_h, self.new_w = self.h + self.pad_b, self.w + self.pad_r
        self.window_h, self.window_w = self.new_h // self.group_size, self.new_w // self.group_size

        self.query_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.motion_q_proj = nn.Linear(d_model, d_model)
        self.motion_k_proj = nn.Linear(d_model, d_model)
        self.motion_gate = nn.Parameter(torch.tensor([0.0]))
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p)

    def _attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)

    def forward(self, x, flow_tokens, t, h=0, w=0):
        bt, n, c = x.shape
        b = bt // t
        c_h = c // self.head
        x = x.view(bt, self.h, self.w, c)
        f = flow_tokens.view(bt, self.h, self.w, c)
        if self.pad_r > 0 or self.pad_b > 0:
            x = F.pad(x, (0, 0, 0, self.pad_r, 0, self.pad_b))
            f = F.pad(f, (0, 0, 0, self.pad_r, 0, self.pad_b))

        q_feat = self.query_embedding(x)
        k_feat = self.key_embedding(x)
        v_feat = self.value_embedding(x)
        q_motion = self.motion_q_proj(f)
        k_motion = self.motion_k_proj(f)
        query = q_feat + self.motion_gate * q_motion
        key = k_feat + self.motion_gate * k_motion
        value = v_feat

        def reshape_to_window(tensor):
            tensor = tensor.view(b, t, self.group_size, self.window_h, self.group_size, self.window_w, self.head, c_h)
            return tensor.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, self.group_size * self.group_size, self.head, -1, c_h)

        query = reshape_to_window(query)
        key = reshape_to_window(key)
        value = reshape_to_window(value)
        att = self._attention(query, key, value)
        att = att.view(b, self.group_size, self.group_size, self.head, t, self.window_h, self.window_w, c_h)
        att = att.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(bt, self.new_h, self.new_w, c)
        if self.pad_b > 0 or self.pad_r > 0:
            att = att[:, :self.h, :self.w, :]
        att = att.reshape(bt, n, c)
        return self.output_linear(att)


class FusionFeedForward(nn.Module):
    def __init__(self, frame_hidden, mlp_ratio, n_vecs, t2t_params, p):
        super().__init__()
        self.kernel_shape = reduce(lambda x, y: x * y, t2t_params["kernel_size"])
        self.t2t_params = t2t_params
        hidden_size = self.kernel_shape * mlp_ratio
        self.conv1 = nn.Linear(frame_hidden, hidden_size)
        self.conv2 = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(p), nn.Linear(hidden_size, frame_hidden), nn.Dropout(p))
        tp = t2t_params.copy()
        self.fold = nn.Fold(**tp)
        if "output_size" in tp:
            del tp["output_size"]
        self.unfold = nn.Unfold(**tp)
        self.n_vecs = n_vecs

    def forward(self, x, n_vecs=0, output_h=0, output_w=0):
        x = self.conv1(x)
        b, n, c = x.size()
        if n_vecs != 0:
            normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, n_vecs, self.kernel_shape).permute(0, 2, 1)
            x_reshaped = x.view(-1, n_vecs, c).permute(0, 2, 1)
            out_fold = F.fold(x_reshaped, output_size=(output_h, output_w), kernel_size=self.t2t_params["kernel_size"], stride=self.t2t_params["stride"], padding=self.t2t_params["padding"])
            norm_fold = F.fold(normalizer, output_size=(output_h, output_w), kernel_size=self.t2t_params["kernel_size"], stride=self.t2t_params["stride"], padding=self.t2t_params["padding"])
            x = self.unfold(out_fold / norm_fold).permute(0, 2, 1).contiguous().view(b, n, c)
        else:
            normalizer = x.new_ones(b, n, self.kernel_shape).view(-1, self.n_vecs, self.kernel_shape).permute(0, 2, 1)
            x_reshaped = x.view(-1, self.n_vecs, c).permute(0, 2, 1)
            x = self.unfold(self.fold(x_reshaped) / self.fold(normalizer)).permute(0, 2, 1).contiguous().view(b, n, c)
        return self.conv2(x)


class TemporalTransformerPatch(nn.Module):
    def __init__(self, token_size, frame_hidden, num_heads, t_group_size, mlp_ratio, dropout, n_vecs, t2t_params):
        super().__init__()
        self.attention = TCTA(token_size=token_size, group_size=t_group_size, d_model=frame_hidden, head=num_heads, p=dropout)
        self.ffn = FusionFeedForward(frame_hidden, mlp_ratio, n_vecs, t2t_params, p=dropout)
        self.norm1 = nn.LayerNorm(frame_hidden)
        self.norm2 = nn.LayerNorm(frame_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, flow_tokens, t, h, w, output_size):
        s = self.norm1(x)
        x = x + self.dropout(self.attention(s, flow_tokens, t, h, w))
        y = self.norm2(x)
        x = x + self.ffn(y, h * w, output_size[0], output_size[1])
        return x
