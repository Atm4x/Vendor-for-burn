from pathlib import Path
import json
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file

class Block(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, hidden)
        self.grn = nn.Module()
        self.grn.gamma = nn.Parameter(torch.zeros(1, 1, hidden))
        self.grn.beta = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pwconv2 = nn.Linear(hidden, dim)

    def forward(self, x):
        y = self.dwconv(x.transpose(1,2)).transpose(1,2)
        y = F.gelu(self.pwconv1(self.norm(y)))
        g = torch.linalg.vector_norm(y, dim=1, keepdim=True)
        n = g / (g.mean(dim=-1, keepdim=True) + 1e-6)
        y = y + self.grn.gamma * y * n + self.grn.beta
        return x + self.pwconv2(y)

class Backbone(nn.Module):
    def __init__(self, dim=8, hidden=16, layers=2):
        super().__init__()
        self.embed = nn.Conv1d(100, dim, 7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList([Block(dim, hidden) for _ in range(layers)])
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, mel):
        x = self.norm(self.embed(mel.transpose(1,2)).transpose(1,2))
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x)

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.head = nn.Module()
        self.head.out = nn.Linear(8,1026)

    def forward(self, mel):
        return self.head.out(self.backbone(mel))

if __name__ == "__main__":
    torch.manual_seed(442)
    network = Toy().eval()
    for name, p in network.named_parameters():
        if name.endswith('norm.weight') or name.endswith('layer_norm.weight'):
            p.data.fill_(1.0)
        elif name.endswith('grn.gamma'):
            p.data.uniform_(-0.1, 0.1)
        else:
            p.data.uniform_(-0.06, 0.06)
    mel = torch.randn(1, 9, 100) * 0.3
    with torch.inference_mode():
        output = network(mel)
        spectral = output[0].T
        magnitude, phase = spectral.chunk(2, dim=0)
        spec = torch.polar(magnitude.exp(), phase)
        spec = torch.cat([spec, spec[:,-1:]], dim=1)
        wave = torch.istft(spec,n_fft=1024,hop_length=256,center=True,window=torch.hann_window(1024))
    root = Path(__file__).resolve().parent.parent/'fixtures'
    save_file({k: v.detach().contiguous() for k,v in network.state_dict().items()}, root/'vocos_decoder_small.safetensors')
    (root/'vocos_decoder_expected.json').write_text(json.dumps({
        'mel':mel.flatten().tolist(),
        'head':output.flatten().tolist(),
        'wave':wave.tolist()
    },separators=(',',':')),encoding='utf-8')
    print('Wrote independent PyTorch Vocos backbone/head/ISTFT fixtures')
