from pathlib import Path
import json
import torch
import torch.nn.functional as F

folder = Path(__file__).resolve().parents[1] / 'fixtures'
case = json.loads((folder/'dit_full_grid_expected.json').read_text(encoding='utf-8'))[2]
batch, frames = case['batch'], case['frames']
mel = case['config']['mel_channels']
length = case['token_length']
noisy = torch.tensor(case['noisy'], dtype=torch.float32).view(batch, frames, mel)
cond = torch.tensor(case['reference'], dtype=torch.float32).view(batch, frames, mel)
tokens = torch.tensor(case['token_ids'], dtype=torch.int32).view(batch, length)
token_ids = F.pad(tokens[:,:frames] + 1, (0,max(0,frames-length)), value=0)
mask = (token_ids != 0).to(torch.int32)
conditional_ids = token_ids.clone()
unconditional_ids = torch.zeros_like(token_ids)
result = {
    'source_batch':batch,'frames':frames,'mel_channels':mel,'token_frames':length,
    'source_noisy':noisy.flatten().tolist(), 'source_condition':cond.flatten().tolist(),
    'source_tokens':tokens.flatten().tolist(),
    'expected_noisy':torch.cat((noisy,noisy),dim=0).flatten().tolist(),
    'expected_condition':torch.cat((cond,torch.zeros_like(cond)),dim=0).flatten().tolist(),
    'expected_tokens':torch.cat((conditional_ids,unconditional_ids),dim=0).flatten().tolist(),
    'expected_text_mask':torch.cat((mask,mask),dim=0).flatten().tolist(),
}
(folder/'trtx_cfg_host_pytorch_expected.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('CFG host packing reference',batch,frames,mel,len(result['expected_tokens']))
