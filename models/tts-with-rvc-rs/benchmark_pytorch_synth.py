import sys,types,importlib,time,json,os,argparse
import torch
parser=argparse.ArgumentParser()
parser.add_argument('--frames',type=int,default=24)
parser.add_argument('--iterations',type=int,default=2)
parser.add_argument('--threads',type=int,default=4)
parser.add_argument('--checkpoint',required=True)
parser.add_argument('--original-root',required=True)
args=parser.parse_args()
torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
root=args.original_root
pkg=types.ModuleType('tts_with_rvc');pkg.__path__=[root+'/tts_with_rvc'];sys.modules['tts_with_rvc']=pkg
for name,rel in [('tts_with_rvc.infer','infer'),('tts_with_rvc.infer.lib','infer/lib'),('tts_with_rvc.infer.lib.infer_pack','infer/lib/infer_pack')]:
 mod=types.ModuleType(name);mod.__path__=[root+'/tts_with_rvc/'+rel];sys.modules[name]=mod
models=importlib.import_module('tts_with_rvc.infer.lib.infer_pack.models')
start=time.perf_counter()
c=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
net=models.SynthesizerTrnMs768NSFsid(*c['config'],is_half=False)
if hasattr(net,'enc_q'):del net.enc_q
status=net.load_state_dict(c['weight'],strict=False)
net=net.cpu().float().eval()
print('load_s',round(time.perf_counter()-start,4),'state_missing',len(status.missing_keys),'unexpected',len(status.unexpected_keys),flush=True)
T=args.frames
rng=torch.Generator().manual_seed(101)
phone=torch.sin(torch.arange(T*768,dtype=torch.float32)*0.013).reshape(1,T,768)*0.1
pitch=torch.full((1,T),128,dtype=torch.long)
pitchf=torch.full((1,T),175.0)
sid=torch.zeros((1,),dtype=torch.long)
length=torch.tensor([T])
results=[]
with torch.inference_mode():
 for i in range(args.iterations+1):
  start=time.perf_counter();g=net.emb_g(sid).unsqueeze(-1)
  start_encoder=time.perf_counter(); m,ls,mask=net.enc_p(phone,pitch,length); ts_enc=time.perf_counter()-start_encoder
  z=m # eliminate random sampling, focus on deterministic modules
  start_flow=time.perf_counter(); z=net.flow(z,mask,g=g,reverse=True); ts_flow=time.perf_counter()-start_flow
  start_decoder=time.perf_counter(); out=net.dec(z*mask,pitchf,g=g); ts_dec=time.perf_counter()-start_decoder
  if i:
   results.append({'encoder_s':round(ts_enc,6),'flow_s':round(ts_flow,6),'nsf_s':round(ts_dec,6),'total_s':round(time.perf_counter()-start,6),'samples':out.shape[-1]})
  print('iteration',i,'warmup' if i==0 else 'bench',results[-1] if i else '',flush=True)
print('RESULT',json.dumps({'framework':'pytorch','model':'CrazyMita','sample_rate':40000,'frames':T,'threads':args.threads,'measurements':results}))
