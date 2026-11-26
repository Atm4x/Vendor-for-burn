from pathlib import Path
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file


class ConvPositionEmbedding(nn.Module):
    def __init__(self, dim, groups):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, 31, padding=15, groups=groups),
            nn.Mish(),
            nn.Conv1d(dim, dim, 31, padding=15, groups=groups),
            nn.Mish(),
        )

    def forward(self, x, mask):
        x = x.transpose(1, 2)
        x = x.masked_fill(~mask[:, None, :], 0)
        for i, layer in enumerate(self.conv1d):
            x = layer(x)
            if i in (0, 2):
                x = x.masked_fill(~mask[:, None, :], 0)
        return x.transpose(1, 2)


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rotary(x):
    length, dim = x.shape[-2:]
    freq = torch.arange(0, dim, 2, dtype=x.dtype) / dim
    inv_freq = 10000 ** (-freq)
    phase = torch.outer(torch.arange(length, dtype=x.dtype), inv_freq)
    full_phase = torch.cat((phase, phase), dim=-1)
    return x * full_phase.cos()[None, None] + rotate_half(x) * full_phase.sin()[None, None]


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0))

    def forward(self, x, mask):
        batch, time, dim = x.shape
        head_dim = dim // self.heads
        shape = lambda y: y.reshape(batch, time, self.heads, head_dim).transpose(1, 2)
        q = rotary(shape(self.to_q(x)))
        k = rotary(shape(self.to_k(x)))
        v = shape(self.to_v(x))
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :], dropout_p=0)
        z = self.to_out(z.transpose(1, 2).reshape(batch, time, dim))
        return z.masked_fill(~mask[:, :, None], 0)


class SpeedPredictorLayer(nn.Module):
    def __init__(self, dim, heads, mult):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.ln1 = nn.LayerNorm(dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(dim, eps=1e-6)
        self.ff = nn.Module()
        self.ff.ff = nn.Sequential(nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(approximate="tanh")), nn.Dropout(0), nn.Linear(dim * mult, dim))

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        return x + self.ff.ff(self.ln2(x))


class SpeedTransformer(nn.Module):
    def __init__(self, dim=8, depth=1, heads=2, mult=2, groups=2):
        super().__init__()
        self.mel_proj = nn.Linear(100, dim)
        self.conv_layer = ConvPositionEmbedding(dim, groups)
        self.transformer_blocks = nn.ModuleList([SpeedPredictorLayer(dim, heads, mult) for _ in range(depth)])
        self.pool = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))
        self.classifier = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 32))

    def forward(self, mel, lens):
        batch, time, _ = mel.shape
        mask = torch.arange(time)[None, :] < lens[:, None]
        x = self.conv_layer(self.mel_proj(mel), mask)
        for block in self.transformer_blocks:
            x = block(x, mask)
        weights = self.pool(x).masked_fill(~mask[:, :, None], -torch.finfo(x.dtype).max)
        x = (x * F.softmax(weights, dim=1)).sum(dim=1)
        return self.classifier(x)


torch.manual_seed(1672)
network = SpeedTransformer().eval()
for name, weight in network.named_parameters():
    if name.endswith("weight") and ("ln" in name or "classifier.0" in name):
        nn.init.uniform_(weight, 0.85, 1.15)
    else:
        nn.init.uniform_(weight, -0.09, 0.09)
mel = torch.randn(2, 5, 100, dtype=torch.float32) * 0.5
lens = torch.tensor([5, 3], dtype=torch.int64)
with torch.inference_mode():
    logits = network(mel, lens)
root = Path(__file__).resolve().parent.parent / "fixtures"
weights = {f"speed_transformer.{name}": tensor.detach().contiguous() for name, tensor in network.state_dict().items()}
save_file(weights, root / "speed_predictor_small.safetensors")
(root / "speed_predictor_expected.json").write_text(json.dumps({
    "dim": 8, "depth": 1, "heads": 2, "ff_mult": 2, "position_groups": 2,
    "batch": 2, "frames": 5, "lengths": lens.tolist(),
    "mel": mel.flatten().tolist(),
    "logits": logits.flatten().tolist(),
    "classes": logits.argmax(dim=-1).tolist()
}, separators=(",", ":")), encoding="utf-8")
print("PyTorch SpeedTransformer fixture saved; classes:", logits.argmax(dim=-1).tolist())
