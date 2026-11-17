"""Independent torch reference for tagged vector-quantize-pytorch 1.14.24 FSQ equations.

The original GroupedResidualFSQ package is not imported in this fixture builder.
This verifies the published index/basis/scaling/project-out formulas only.
"""
from __future__ import annotations
import pathlib
import torch

LEVELS = [8,5,5,5]


def weights(n: int, seed: int) -> torch.Tensor:
    return torch.tensor([((i*13+seed)%37)/40.0-0.45 for i in range(n)],dtype=torch.float32)


def bias(n: int, seed: int) -> torch.Tensor:
    return torch.tensor([((i*7+seed)%13)/30.0-0.2 for i in range(n)],dtype=torch.float32)


def run(batch: int, groups: int, q: int, frames: int, group_width: int, seed: int, dropout: bool):
    raw = torch.tensor([(i*37+seed)%1000 for i in range(batch*groups*q*frames)],dtype=torch.int32)
    if dropout:
        raw[3::4] = -1
    raw = raw.view(batch, groups*q, frames)
    basis = torch.tensor([1,8,40,200],dtype=torch.int32)
    levels = torch.tensor(LEVELS,dtype=torch.int32)
    x = torch.zeros((batch,groups*group_width,frames),dtype=torch.float32)
    for g in range(groups):
        for t in range(frames):
            for b in range(batch):
                residual = torch.zeros(len(LEVELS),dtype=torch.float32)
                for qi in range(q):
                    code = int(raw[b,g*q+qi,t])
                    if code==-1: continue
                    digits = (code//basis)%levels
                    normalized = (digits.float()-(levels//2).float())/(levels//2).float()
                    residual += normalized * (levels.float()-1).pow(-qi)
                proj=weights(group_width*len(LEVELS),seed+g).view(group_width,len(LEVELS))
                projected = torch.mv(proj,residual)+bias(group_width,seed+g+1)
                x[b,g*group_width:(g+1)*group_width,t]=projected
    return raw,x


def main() -> None:
    lines=["# synthetic FP32 formulas derived from 1.14.24 ResidualFSQ / GroupedResidualFSQ; package itself not executed"]
    for name, batch, groups, q, frames, width, seed, dropout in [
        ('fsq_8group_q1',1,8,1,3,8,3,False),
        ('fsq_2group_q3',2,2,3,4,3,7,True),
    ]:
        codes,x=run(batch,groups,q,frames,width,seed,dropout)
        lines.append(name + ' ' + f'{batch},{groups},{q},{frames},{width},{seed},{int(dropout)}' + ' '
                     + ','.join(map(str,codes.flatten().tolist())) + ' '
                     + ' '.join(f'{v:.9g}' for v in x.flatten().tolist()))
    path=pathlib.Path(__file__).with_suffix('.txt')
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('created',path)


if __name__=='__main__':
    main()
