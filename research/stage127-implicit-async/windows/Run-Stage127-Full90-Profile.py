#!/usr/bin/env python3
"""Run full ~90s RVC through Stage127 implicit GEMM sync/async, NTC hybrid and original PyTorch CUDA."""
import argparse
import array
import csv
import hashlib
import json
import math
import re
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import wave
import zipfile
from datetime import datetime,timezone

FLAGS=("RVC_NSF_CONV1D_GEMM_ASYNC","RVC_NSF_RESBLOCK_FP16_MIN_CHANNELS","RVC_NSF_RESBLOCK_TRANSPORT","RVC_NSF_RESBLOCK_SHARED_NTC","RVC_NSF_RESBLOCK_TEMPORAL_TILE","RVC_NSF_RESBLOCK_ACTIVATION_PLACEMENT","RVC_NSF_RESBLOCK_LINEAR_INTERIOR","RVC_NSF_RESBLOCK_RESIDUE_ORDER","RVC_NSF_RESBLOCK_PLANE_BROADCAST","RVC_NSF_RESBLOCK_WEIGHT_PREFETCH","RVC_RMVPE_GRU_PERSISTENT","RVC_NSF_IMPLICIT_GEMM","RVC_NSF_CONV1D_GEMM","RVC_NSF_TRANSPOSE_GEMM","RVC_NSF_CONV_KERNEL","RVC_NSF_RESBLOCK_FP16","RVC_NSF_PAIR_SHARED_TILE","RVC_NSF_FUSED_PAIR_MAX_CHANNELS","RVC_RMVPE_UNET_GEMM","RVC_RMVPE_UNET_COMPENSATED_TF32","RVC_RMVPE_UNET_COMPENSATED_BATCH_TF32","RVC_RMVPE_UNET_WINOGRAD","RVC_HUBERT_FRONTEND_GEMM","RVC_RMVPE_UNET_SPATIAL_QUAD","RVC_RMVPE_UNET_SPATIAL_REUSE","RVC_GPU_FAISS","RVC_GPU_F0","RVC_GLOBAL_F0","RVC_PROFILE","RVC_DIAGNOSTIC_TRACE","RVC_NVTX","RVC_CHUNK_STRATEGY","RVC_QUIET_EXTRA_CHUNKS","RVC_CHUNK_BATCH_SCHEDULE","RVC_F0_REFERENCE_DIR","RVC_F0_REFERENCE_MODE","RVC_NSF_GLOBAL_PHASE","RVC_NSF_GLOBAL_NOISE","RVC_RMVPE_MEL_REUSE")
EXPECTED_SOURCE={"tts_with_rvc/infer/vc/pipeline.py":"8e25cddb6d9a616a1d2f9911146e2a1d1c77900d40421c8d17ebeea16ce1736e","tts_with_rvc/infer/vc/modules.py":"4d3bada97629b2a61a1497cd0c8f92a707e78a355c1dbc2729556ec70fee926a","tts_with_rvc/infer/vc/config.py":"4470db3c5f9b1d3542cf8fdfcdc02d658be2368b10d5dd12c79f16dd1a15c82f","tts_with_rvc/infer/lib/rmvpe.py":"da5d5d311f220a18164b47ce08a48baebf93f4ab","tts_with_rvc/infer/lib/infer_pack/models.py":"9a642acdad8edf497afb5afc76ce0c722564a04e9d049605e1096d2585fffa4e"}
def abs_path(s,base):
    p=Path(s)
    return (p if p.is_absolute() else base/p).resolve()
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def metadata(p):
    with wave.open(str(p),"rb") as f:
        return dict(rate=f.getframerate(),channels=f.getnchannels(),sample_width=f.getsampwidth(),samples=f.getnframes(),seconds=f.getnframes()/f.getframerate())
def pcm(left,right):
    with wave.open(str(left),"rb") as a,wave.open(str(right),"rb") as b:
        if (a.getnchannels(),a.getsampwidth(),a.getframerate(),a.getnframes())!=(b.getnchannels(),b.getsampwidth(),b.getframerate(),b.getnframes()) or a.getnchannels()!=1 or a.getsampwidth()!=2:return dict(compatible=False)
        different=maximum=error=signal=total=0
        while True:
            x=a.readframes(262144);y=b.readframes(262144)
            if not x and not y:break
            if len(x)!=len(y):return dict(compatible=False,truncated=True)
            av=array.array("h");bv=array.array("h");av.frombytes(x);bv.frombytes(y)
            if sys.byteorder!="little":av.byteswap();bv.byteswap()
            for u,v in zip(av,bv):
                d=int(u)-int(v);total+=1;different+=d!=0;maximum=max(maximum,abs(d));error+=d*d;signal+=int(u)*int(u)
    return dict(compatible=True,samples=total,different_samples=different,max_abs_pcm_lsb=maximum,rmse_pcm_lsb=math.sqrt(error/total),snr_db=10*math.log10(signal/error) if error and signal else ("infinite" if not error else None))
MODES=("implicit_sync","implicit_async","hybrid_sync","ntc_combined","implicit_fp16")
def native_env(mode):
    if mode not in MODES:raise ValueError("Unknown mode "+mode)
    env=os.environ.copy()
    for k in FLAGS:env.pop(k,None)
    env["RVC_RMVPE_GRU_PERSISTENT"]="1"
    if mode.startswith("implicit"):
        env["RVC_NSF_IMPLICIT_GEMM"]="1"
        if mode=="implicit_async":env["RVC_NSF_CONV1D_GEMM_ASYNC"]="1"
        if mode=="implicit_fp16":
            env["RVC_NSF_RESBLOCK_FP16"]="1"
            env["RVC_NSF_RESBLOCK_FP16_MIN_CHANNELS"]="64"
    else:
        env.update(RVC_NSF_RESBLOCK_TRANSPORT="1",RVC_NSF_RESBLOCK_SHARED_NTC="1",RVC_NSF_RESBLOCK_TEMPORAL_TILE="4",RVC_NSF_RESBLOCK_ACTIVATION_PLACEMENT="1",RVC_NSF_TRANSPOSE_GEMM="1" if mode.startswith("hybrid") else "0")
        if mode.endswith("async"):env["RVC_NSF_CONV1D_GEMM_ASYNC"]="1"
        if mode=="ntc_combined":
            env["RVC_NSF_RESBLOCK_LINEAR_INTERIOR"]="1"
            env["RVC_NSF_RESBLOCK_RESIDUE_ORDER"]="1"
    return env

def validate_native(x,mode,args,seconds):
    flags=x.get("nsf_resblock_benchmark_requested")
    transport=mode.startswith("hybrid") or mode=="ntc_combined"
    expected=dict(transport=transport,shared_ntc=transport,temporal_tile="4" if transport else None,activation_placement=transport,linear_interior=mode=="ntc_combined",residue_order=mode=="ntc_combined",plane_broadcast=False,weight_prefetch=False)
    if not isinstance(flags,dict) or any(flags.get(k)!=v for k,v in expected.items()):raise ValueError("Stage127 route flags wrong for "+mode)
    expect_conv="direct" if mode=="ntc_combined" else "implicit_gemm" if mode.startswith("implicit") else "mixed"
    # NTC hybrid is split: ResBlock uses transport, ConvTranspose uses GEMM.
    if mode.startswith("hybrid"):expect_conv="mixed"
    expect_loading="async_cyclic" if mode.endswith("async") else "sync_cyclic"
    if (x.get("backend")!="cuda" or x.get("graph_replay") is not False or x.get("nsf_conv_execution")!=expect_conv
        or x.get("nsf_conv1d_gemm_loading")!=expect_loading or x.get("nsf_execution")!="sequential"
        or x.get("rmvpe_gru_execution")!="persistent" or x.get("core_frames")!=args.core_frames
        or x.get("tensor_batch_size")!=args.batch_size or x.get("iterations_exact_f32") is not True
        or len(x.get("inference_ms",[]))!=args.runs or not x.get("chunk_counts")
        or abs(x.get("source_seconds",0)-seconds)>0.03):raise ValueError("Full-waveform policy/iteration mismatch "+mode)
    expect_precision="fp16_island_fp32_stage" if mode=="implicit_fp16" else "fp32"
    if x.get("nsf_resblock_precision")!=expect_precision:raise ValueError("Wrong ResBlock precision for "+mode)
    if x.get("nsf_transpose_execution")!=("gemm" if (mode.startswith("implicit") or mode.startswith("hybrid")) else "direct"):
        raise ValueError("Wrong ConvTranspose route for "+mode)

def nsys_target_name(out,mode):
    candidates=sorted(out.glob(mode+".nsys.summary*cuda_gpu_kern_sum*.csv"))
    if not candidates:candidates=sorted(out.glob(mode+".nsys.summary*.csv"))
    for path in candidates:
        try:
            with path.open(encoding="utf-8-sig",newline="") as handle:
                rows=list(csv.DictReader(handle))
            if not rows:continue
            keys=rows[0].keys()
            name_col=next((k for k in keys if k.lower().strip()=="name"),None)
            time_col=next((k for k in keys if "total" in k.lower() and "time" in k.lower()),None)
            if name_col is None or time_col is None:continue
            ranked=sorted(rows,key=lambda row:float(re.sub(r"[^0-9.]","",row.get(time_col,"") or "0") or 0),reverse=True)
            for row in ranked:
                name=row.get(name_col,"").strip()
                if not name:continue
                if mode=="ntc_combined" or mode.startswith("hybrid"):
                    if "preact_residual_temporal_kernel" not in name:continue
                elif not any(token in name.lower() for token in ("conv","gemm","matmul","cmma")):
                    continue
                # Match the demangled kernel family without depending on argument-template text.
                candidate=name.split("(",1)[0].split("<",1)[0].strip()
                if len(candidate)>100:candidate=candidate[-100:]
                if len(candidate)<8:continue
                return re.escape(candidate),str(path),name
        except (OSError,ValueError,TypeError):continue
    return None,None,None

def run(name,cmd,out,cwd,env=None):
    print("Stage127:",name,flush=True)
    start=time.perf_counter();p=subprocess.run(cmd,cwd=str(cwd),env=env,capture_output=True,check=False)
    (out/(name+".stdout.log")).write_bytes(p.stdout);(out/(name+".stderr.log")).write_bytes(p.stderr)
    if p.returncode:print("FAILED",name,p.returncode,p.stderr.decode("utf-8","replace")[-1000:],file=sys.stderr)
    return p.returncode,(time.perf_counter()-start)*1000
def archive(src,dst):
    with zipfile.ZipFile(dst,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in src.rglob("*"):
            if p.is_file() and p.suffix.lower() not in (".wav",".pt",".pth",".index",".flac",".mp3",".onnx",".safetensors",".exe",".dll"):z.write(p,p.relative_to(src))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rust-exe",default="tts-with-rvc-rs.exe")
    p.add_argument("--torch-python",default=r"D:\Projects\tts-with-rvc\.venv\Scripts\python.exe")
    p.add_argument("--original-root",default=r"D:\Projects\tts-with-rvc")
    p.add_argument("--assets-dir",default="embed")
    p.add_argument("--input-wav",default="MitaKind.wav")
    p.add_argument("--voice",default="CrazyMita.pth");p.add_argument("--hubert",default="hubert_base.pt");p.add_argument("--rmvpe",default="rmvpe.pt");p.add_argument("--index",default="CrazyMita.index")
    p.add_argument("--index-rate",type=float,default=0.75);p.add_argument("--core-frames",type=int,default=300);p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--cycles",type=int,default=2);p.add_argument("--runs",type=int,default=3);p.add_argument("--modes",default="implicit_sync,implicit_async,hybrid_sync,ntc_combined")
    p.add_argument("--torch-precision",choices=("both","fp32","fp16"),default="both")
    p.add_argument("--skip-torch",action="store_true");p.add_argument("--check-only",action="store_true");p.add_argument("--keep-wav",action="store_true");p.add_argument("--out-dir",default="stage127_full90_results")
    p.add_argument("--profiler",choices=("none","nsys","ncu","all"),default="all")
    p.add_argument("--profile-modes",default="implicit_sync,implicit_async,ntc_combined")
    p.add_argument("--ncu-exe",default=r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe")
    p.add_argument("--nsys-exe",default=r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.6.2\target-windows-x64\nsys.exe")
    p.add_argument("--nvtx-dll-dir",default=r"C:\Users\mrtak\miniconda3\envs\torch2.6.0\Lib\site-packages\torch\lib")
    p.add_argument("--ncu-kernel-regex",default="")
    p.add_argument("--ncu-launch-skip",type=int,default=0)
    p.add_argument("--ncu-launch-count",type=int,default=4)
    p.add_argument("--ncu-start-seconds",type=float,default=0.0)
    p.add_argument("--ncu-duration-seconds",type=float,default=3.0)
    p.add_argument("--torch-nsys",action="store_true")
    p.add_argument("--max-pcm-lsb",type=int,default=128)
    p.add_argument("--min-snr-db",type=float,default=75.0)
    args=p.parse_args();here=Path(__file__).resolve().parent
    selected=[x.strip() for x in args.profile_modes.split(",") if x.strip()]
    if any(x not in MODES for x in selected) or len(selected)!=len(set(selected)):
        p.error("Invalid profile modes")
    if (args.ncu_launch_skip<0 or not 1<=args.ncu_launch_count<=20 or args.ncu_duration_seconds>9
        or args.ncu_duration_seconds<=0 or args.max_pcm_lsb<0 or args.min_snr_db<=0):p.error("Invalid profiler/quality parameters")
    modes=[x.strip().lower() for x in args.modes.split(",") if x.strip()]
    if not (2<=args.runs<=16 and 1<=args.cycles<=6 and 25<=args.core_frames<=900 and args.core_frames+84<=1000 and 1<=args.batch_size<=32 and 0<=args.index_rate<=1 and len(modes)==len(set(modes)) and "implicit_sync" in modes and all(x in MODES for x in modes)):p.error("invalid geometry, repeats or modes")
    assets=abs_path(args.assets_dir,here);exe=abs_path(args.rust_exe,here);python=abs_path(args.torch_python,here);original=abs_path(args.original_root,here)
    paths={name:abs_path(getattr(args,name),assets) for name in ("voice","hubert","rmvpe","input_wav","index")}
    files=[exe,paths["voice"],paths["hubert"],paths["rmvpe"],paths["input_wav"]]
    if args.index_rate:files.append(paths["index"])
    if not args.skip_torch:files.extend([python,original/"tts_with_rvc/infer/vc/pipeline.py"])
    for file in files:
        if not file.is_file():p.error("missing required file: "+str(file))
    wave_input=metadata(paths["input_wav"])
    if not 85<=wave_input["seconds"]<=95:p.error("Requires actual 85–95 second WAV; no automatic truncation. Supply --input-wav full recording.")
    if args.profiler!="none" and not args.check_only:
        needed=[args.ncu_exe] if args.profiler in ("ncu","all") else []
        if args.profiler in ("nsys","all"):needed+=[args.nsys_exe]
        for executable in needed:
            if not Path(executable).is_file():p.error("Missing profiler "+executable)
    if args.check_only:
        print(json.dumps({"preflight":"PASS","wav":wave_input,"modes":modes,"torch":not args.skip_torch,"profile_modes":selected,"profiler":args.profiler},indent=2));return 0
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ");root=abs_path(args.out_dir,here);out=root/stamp;out.mkdir(parents=True,exist_ok=False)
    fp={rel:{"expected":old,"actual":hashlib.sha256((original/rel).read_bytes().replace(b"\r\n",b"\n")).hexdigest() if (original/rel).is_file() else None} for rel,old in EXPECTED_SOURCE.items()} if not args.skip_torch else {}
    if fp and any(x["actual"]!=x["expected"] for x in fp.values()):print("WARNING: installed reference source differs from Stage80 pinned original; SHA logged")
    checks={k:sha(v) for k,v in paths.items() if k!="index" or args.index_rate};checks["rust_exe"]=sha(exe)
    prec=[] if args.skip_torch else (["fp32","fp16"] if args.torch_precision=="both" else [args.torch_precision])
    tasks=["rust_"+x for x in modes]
    for i,v in enumerate(prec):tasks.insert(min(len(tasks),2*i+1),"torch_"+v)
    rows=[];failures=[];parity=[];profiles=[]
    summary=dict(stage=127,status="INCOMPLETE",source_wav=wave_input,input_sha256=checks,torch_source_fingerprints=fp,benchmark_note="Rust synchronized resident infer_long_filtered_16khz_batched vs original PyTorch full pipeline have different input processing, F0, chunk schedules and timing scope. Compare same-Rust routes causally; PyTorch ratios are product-level only.",native_geometry={"core_frames":args.core_frames,"batch_size":args.batch_size},cycles=args.cycles,runs=args.runs,results=rows,rust_parity=parity,failures=failures,profiles=profiles,profiled_scope="NSYS full 90s; NCU 3s real source segment with kernel filter and strict hardware replay separation")
    try:
        try:
            q=subprocess.run(["nvidia-smi","--query-gpu=name,driver_version,memory.total","--format=csv,noheader"],capture_output=True,timeout=20,check=False)
            (out/"nvidia-smi.txt").write_bytes(q.stdout+q.stderr)
        except (OSError,subprocess.TimeoutExpired):pass
        for cycle in range(1,args.cycles+1):
            for task in (tasks if cycle%2 else list(reversed(tasks))):
                name=f"c{cycle}_{task}"
                if task.startswith("rust_"):
                    mode=task[5:];wav=out/(name+".wav")
                    command=[str(exe),"infer-wav-full-benchmark",str(paths["voice"]),str(paths["hubert"]),str(paths["rmvpe"]),str(paths["index"]) if args.index_rate else "-",str(paths["input_wav"]),str(wav),str(args.index_rate),"cuda",str(args.runs),str(args.core_frames),str(args.batch_size)]
                    rc,wall=run(name,command,out,here,native_env(mode));row=dict(cycle=cycle,kind="rust",mode=mode,exit_code=rc,process_wall_ms=wall)
                    if rc:failures.append({"case":name,"error":"Rust process failed"})
                    else:
                        try:
                            j=json.loads(Path(str(wav)+".benchmark.json").read_text(encoding="utf-8-sig"));validate_native(j,mode,args,wave_input["seconds"])
                            w=metadata(wav)
                            if (w["rate"],w["channels"],w["sample_width"],w["samples"])!=(40000,1,2,j["output_samples"]):raise ValueError("invalid Rust PCM16 output")
                            row.update(warm_median_ms=j["warm_median_ms"],warm_rtf=j["warm_rtf"],inference_ms=j["inference_ms"],first_ms=j["first_inference_ms"],chunk_counts=j["chunk_counts"],source_seconds=j["source_seconds"],output_samples=j["output_samples"],sha256=sha(wav),wav_path=str(wav))
                        except (OSError,ValueError,KeyError,TypeError) as e:failures.append({"case":name,"error":str(e)})
                else:
                    precision=task[6:];target=out/name
                    command=[str(python),str(here/"Run-Stage127-OriginalTorch.py"),"--original-root",str(original),"--assets-dir",str(assets),"--input-wav",str(paths["input_wav"]),"--voice",paths["voice"].name,"--hubert",paths["hubert"].name,"--rmvpe",paths["rmvpe"].name,"--index",paths["index"].name,"--index-rate",str(args.index_rate),"--runs",str(args.runs),"--precision",precision,"--output",str(target)]
                    if precision=="fp32":command.append("--strict-fp32")
                    rc,wall=run(name,command,out,here);row=dict(cycle=cycle,kind="torch",mode=precision,exit_code=rc,process_wall_ms=wall)
                    if rc:failures.append({"case":name,"error":"Original PyTorch process failed"})
                    else:
                        try:
                            j=json.loads((target/"summary.json").read_text(encoding="utf-8-sig"))
                            if j["precision"]!=precision or len(j["inference_ms"])!=args.runs or abs(j["source_seconds"]-wave_input["seconds"])>.1:raise ValueError("PyTorch input/repetitions mismatch")
                            row.update(warm_median_ms=j["warm_median_ms"],warm_rtf=j["warm_rtf"],inference_ms=j["inference_ms"],output_samples=j["output_samples"],sha256=j["sha256"],device=j["device"],cudnn_tf32=j["cudnn_allow_tf32"])
                        except (OSError,ValueError,KeyError,TypeError) as e:failures.append({"case":name,"error":str(e)})
                rows.append(row)
        for cycle in range(1,args.cycles+1):
            control=next((r for r in rows if r["cycle"]==cycle and r["kind"]=="rust" and r["mode"]=="implicit_sync" and "wav_path" in r),None)
            if control is None:failures.append({"case":f"c{cycle}_control","error":"missing control WAV"});continue
            for r in rows:
                if r["cycle"]!=cycle or r["kind"]!="rust" or r["mode"]=="implicit_sync" or "wav_path" not in r:continue
                d=pcm(Path(control["wav_path"]),Path(r["wav_path"]));parity.append(dict(cycle=cycle,mode=r["mode"],**d))
                if not d.get("compatible") or d["max_abs_pcm_lsb"]>args.max_pcm_lsb or (isinstance(d.get("snr_db"),(int,float)) and d["snr_db"]<args.min_snr_db):
                    failures.append({"case":f"c{cycle}_{r['mode']}","error":"REJECT_PCM_QUALITY",**d})
        aggregate={}
        for name in sorted({f"{r['kind']}_{r['mode']}" for r in rows if "warm_median_ms" in r}):
            values=[r["warm_median_ms"] for r in rows if f"{r['kind']}_{r['mode']}"==name and "warm_median_ms" in r]
            aggregate[name]=dict(cycles=len(values),median_warm_ms=statistics.median(values),range_ms=[min(values),max(values)])
        summary["aggregate"]=aggregate
        if "rust_implicit_sync" in aggregate:
            summary["torch_over_rust_implicit_sync_ratio_non_equivalent"]={name:round(data["median_warm_ms"]/aggregate["rust_implicit_sync"]["median_warm_ms"],6) for name,data in aggregate.items() if name.startswith("torch_")}
        rejected_modes={f.get("case","").rsplit("_",1)[-1] for f in failures if f.get("error")=="REJECT_PCM_QUALITY"}
        if args.profiler!="none":
            nvtx_dir=Path(args.nvtx_dll_dir)
            nvtx_env=os.environ.copy()
            if nvtx_dir.joinpath("nvToolsExt64_1.dll").is_file():
                nvtx_env["PATH"]=str(nvtx_dir)+os.pathsep+nvtx_env.get("PATH","")
            else:
                failures.append({"case":"nvtx","error":"NVTX DLL not found; kernel-name NCU still usable; NSYS NVTX scope unavailable"})
            for mode in selected:
                if mode in rejected_modes:
                    profiles.append(dict(mode=mode,type="skipped_quality",reason="PCM quality gate failed"))
                    continue
                if not any(r.get("kind")=="rust" and r.get("mode")==mode and "warm_median_ms" in r for r in rows):continue
                env=native_env(mode);env["RVC_NVTX"]="1"
                env["PATH"]=nvtx_env.get("PATH",env.get("PATH",""))
                def full_command(wav,runs):
                    return [str(exe),"infer-wav-full-benchmark",str(paths["voice"]),str(paths["hubert"]),str(paths["rmvpe"]),str(paths["index"]) if args.index_rate else "-",str(paths["input_wav"]),str(wav),str(args.index_rate),"cuda",str(runs),str(args.core_frames),str(args.batch_size)]
                if args.profiler in ("nsys","all"):
                    stem=str(out/(mode+".nsys"))
                    cmd=[str(Path(args.nsys_exe)),"profile","--trace=cuda,nvtx","--sample=none","--force-overwrite=true","--output",stem,*full_command(out/(mode+".nsys.wav"),2)]
                    rc,_=run(mode+".nsys",cmd,out,here,env)
                    profiles.append(dict(mode=mode,type="nsys",exit_code=rc,report=stem+".nsys-rep"))
                    if rc or not Path(stem+".nsys-rep").is_file():failures.append({"case":mode+".nsys","error":"NSYS profiling failed"})
                    else:
                        rc,_=run(mode+".nsys-export",[str(Path(args.nsys_exe)),"export","--type","sqlite","--force-overwrite=true","--output",stem+".sqlite",stem+".nsys-rep"],out,here,env)
                        if rc:failures.append({"case":mode+".nsys-export","error":"NSYS SQLite export failed"})
                        rc,_=run(mode+".nsys-summary",[str(Path(args.nsys_exe)),"stats","--report","cuda_gpu_kern_sum","--format","csv","--output",stem+".summary",stem+".nsys-rep"],out,here,env)
                        if rc:failures.append({"case":mode+".nsys-summary","error":"NSYS kernel summary export failed; raw report kept"})
                if args.profiler in ("ncu","all"):
                    stem=str(out/(mode+".ncu"))
                    auto_regex,auto_source,auto_name=nsys_target_name(out,mode)
                    regex=args.ncu_kernel_regex or auto_regex or ("preact_residual_temporal_kernel" if mode.startswith("hybrid") or mode=="ntc_combined" else "conv")
                    cmd=[str(Path(args.ncu_exe)),"--target-processes","all","--kernel-name","regex:"+regex,"--launch-skip",str(args.ncu_launch_skip),"--launch-count",str(args.ncu_launch_count),"--set","full","--force-overwrite","--export",stem+".ncu-rep",str(exe),"infer-wav-segment-benchmark",str(paths["voice"]),str(paths["hubert"]),str(paths["rmvpe"]),str(paths["index"]) if args.index_rate else "-",str(paths["input_wav"]),stem+".wav",str(args.ncu_start_seconds),str(args.ncu_duration_seconds),str(args.index_rate),"0","cuda","2"]
                    rc,_=run(mode+".ncu",cmd,out,here,env)
                    profiles.append(dict(mode=mode,type="ncu",exit_code=rc,regex=regex,source_nsys_summary=auto_source,auto_selected_kernel=auto_name,launch_skip=args.ncu_launch_skip,launch_count=args.ncu_launch_count,report=stem+".ncu-rep"))
                    if rc or not Path(stem+".ncu-rep").is_file():failures.append({"case":mode+".ncu","error":"NCU no report; inspect per-kernel logs"})
                    else:
                        exporter=here/"Export-Stage127-NCU-Metrics.py"
                        rc,_=run(mode+".ncu-export",[str(python),str(exporter),stem+".ncu-rep",stem+".csv"],out,here,env)
                        if rc:failures.append({"case":mode+".ncu-export","error":"NCU CSV conversion failed; raw report retained"})
                        elif "No kernels were profiled" in (out/(mode+".ncu.stderr.log")).read_text(errors="replace"):
                            failures.append({"case":mode+".ncu","error":"No target CUDA kernels captured; adjust launch skip/kernel regex"})
            if args.torch_nsys and not args.skip_torch and args.profiler in ("nsys","all"):
                precision="fp16" if args.torch_precision=="both" else args.torch_precision
                stem=str(out/("torch_"+precision+".nsys"))
                cmd=[str(Path(args.nsys_exe)),"profile","--trace=cuda,nvtx","--sample=none","--force-overwrite=true","--output",stem,str(python),str(here/"Run-Stage127-OriginalTorch.py"),"--original-root",str(original),"--assets-dir",str(assets),"--input-wav",str(paths["input_wav"]),"--voice",paths["voice"].name,"--hubert",paths["hubert"].name,"--rmvpe",paths["rmvpe"].name,"--index",paths["index"].name,"--index-rate",str(args.index_rate),"--runs","2","--precision",precision,"--output",str(out/("torch_"+precision+".nsys_aux"))]
                if precision=="fp32":cmd.append("--strict-fp32")
                rc,_=run("torch_"+precision+".nsys",cmd,out,here,nvtx_env)
                profiles.append(dict(mode="torch_"+precision,type="nsys",exit_code=rc,report=stem+".nsys-rep"))
                if rc:failures.append({"case":"torch_"+precision+".nsys","error":"PyTorch NSYS capture failed"})
                else:
                    run("torch_"+precision+".nsys-export",[str(Path(args.nsys_exe)),"export","--type","sqlite","--force-overwrite=true","--output",stem+".sqlite",stem+".nsys-rep"],out,here,nvtx_env)
                    run("torch_"+precision+".nsys-summary",[str(Path(args.nsys_exe)),"stats","--report","cuda_gpu_kern_sum","--format","csv","--output",stem+".summary",stem+".nsys-rep"],out,here,nvtx_env)
        if profiles:
            rc,_=run("nsight-analysis",[str(python),str(here/"Analyze-Stage127-Nsight.py"),str(out)],out,here)
            if rc:failures.append({"case":"nsight-analysis","error":"Analysis failed; raw profiler reports retained"})
        summary["status"]="PASS" if not failures else "FAIL"
    finally:
        (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        with (out/"summary.csv").open("w",newline="",encoding="utf-8") as f:
            fields=("cycle","kind","mode","exit_code","warm_median_ms","warm_rtf","first_ms","process_wall_ms","output_samples")
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for r in rows:writer.writerow({k:r.get(k,"") for k in fields})
        archive(out,root/(stamp+".zip"))
        print("Stage127 diagnostic ZIP:",root/(stamp+".zip"),flush=True)
        if not args.keep_wav:
            for w in out.rglob("*.wav"):w.unlink()
    for k,v in summary.get("aggregate",{}).items():print(k,v["median_warm_ms"],"ms")
    print("Stage127",summary["status"],"failures",len(failures))
    return int(bool(failures))
if __name__=="__main__":raise SystemExit(main())
