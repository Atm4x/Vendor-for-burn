from pathlib import Path
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parents[1] / 'core/fixtures/trtx_pointwise_original_pytorch.txt'
torch.manual_seed(126)
N, C, T = 2, 5, 7
x = torch.randn(N, C, T, dtype=torch.float32)
gamma = torch.randn(C)
beta = torch.randn(C)
y = torch.randn(N, C, T)
z = torch.randn(N, C, T)

def line(label, t):
    return label + ' ' + ' '.join(f'{v:.10g}' for v in t.contiguous().flatten().tolist())

cases = [
 ('gelu_erf', x, torch.nn.GELU()(x), torch.empty(0),torch.empty(0)),
 ('silu', x, F.silu(x), torch.empty(0),torch.empty(0)),
 ('tanh', x, x.tanh(),torch.empty(0),torch.empty(0)),
 ('layer_scale', x, x*gamma[None,:,None], gamma,torch.empty(0)),
 ('layernorm_last',x,F.layer_norm(x.transpose(1,2),(C,),gamma,beta,1e-6).transpose(1,2),gamma,beta),
 ('layernorm_first',x,((x-x.mean(1,keepdim=True))/torch.sqrt(((x-x.mean(1,keepdim=True))**2).mean(1,keepdim=True)+1e-6))*gamma[None,:,None]+beta[None,:,None], gamma,beta),
 ('add',x,x+y,torch.empty(0),y),
 ('parallel_mean',x,torch.stack((x,y,z)).mean(0),y,z),
]
lines=[]
for name, values,expected, w,b in cases:
    lines.extend([f'case {name} {N} {C} {T}',line('input',values),line('extra1',w),line('extra2',b),line('output',expected)])
OUT.write_text('\n'.join(lines)+'\n', encoding='utf8')
print(f'{len(cases)} PyTorch pointwise equation fixtures written: {OUT}')
