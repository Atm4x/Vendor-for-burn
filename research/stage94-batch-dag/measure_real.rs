#[path = "../../models/tts-with-rvc-rs/src/audio_io.rs"] mod audio_io;
#[path = "../../models/tts-with-rvc-rs/src/preprocess.rs"] mod preprocess;
#[path = "../../models/tts-with-rvc-rs/src/chunk_plan.rs"] mod chunk_plan;
#[path = "../../models/tts-with-rvc-rs/src/chunked.rs"] mod chunked;

use chunk_plan::{BatchSchedule, ChunkStrategy, CostModel, plan_tensor_batches};
use chunked::{planned_chunk_ranges_for_batch, ChunkConfig};

fn report(pcm:&[f32], name:&str, config:ChunkConfig, strategy:ChunkStrategy, batch_size:usize, model:CostModel) {
    let plan=planned_chunk_ranges_for_batch(pcm,config,strategy,batch_size).unwrap();
    let contiguous=plan_tensor_batches(&plan,batch_size,BatchSchedule::Contiguous).unwrap();
    let gather=plan_tensor_batches(&plan,batch_size,BatchSchedule::ShapeGather{lookahead:64}).unwrap();
    let mut shapes=std::collections::BTreeMap::<usize,usize>::new();
    let mut work=0.0;
    let mut area=0u64;
    let mut frames=0usize;
    for &(start,end) in &plan {
        let size=end-start;
        let length=size+2*config.context_frames+config.trailing_frames;
        *shapes.entry(size).or_default()+=1;
        frames+=length;
        area+=length as u64*length as u64;
        work+=model.linear as f64*length as f64 +model.attention_milli as f64/1000.0*(length*length) as f64;
    }
    let mut total_rms=0.0f64;
    let energy_norm=(pcm.iter().map(|x|(*x as f64).powi(2)).sum::<f64>()/pcm.len() as f64).max(1e-20);
    let mut seam_cost=0.0;
    for &(_,end) in &plan[..plan.len()-1] {
        let left=(end*160).saturating_sub(160);
        let right=(end*160+160).min(pcm.len());
        let e=pcm[left..right].iter().map(|x|(*x as f64).powi(2)).sum::<f64>()/(right-left) as f64;
        total_rms+=e;
        seam_cost+= (e/energy_norm).min(10_000.0)*model.splice as f64;
    }
    let distinct=shapes.len();
    let largest=shapes.values().copied().max().unwrap_or(0);
    let proxy=work+seam_cost+gather.len() as f64*model.launch as f64;
    println!("{{\"name\":\"{name}\",\"chunks\":{},\"batch_size\":{batch_size},\"contiguous_calls\":{},\"gather_calls\":{},\"shapes\":{distinct},\"largest_bucket\":{largest},\"input_seconds\":{:.3},\"attention_area\":{area},\"avg_boundary_energy\":{:.10},\"batch_proxy\":{:.3},\"boundaries\":{:?}}}",plan.len(),contiguous.len(),gather.len(),frames as f64/100.0,total_rms/(plan.len()-1).max(1) as f64,proxy,plan.iter().take(plan.len()-1).map(|p|p.1).collect::<Vec<_>>());
}

fn main() -> Result<(),String> {
    let input=std::env::args().nth(1).ok_or("usage: stage94_measure input.wav".to_owned())?;
    let wave=audio_io::read_mono_wav(&input)?;
    let pcm=audio_io::resample(&wave.samples,wave.sample_rate,16_000)?;
    let pcm=preprocess::highpass_48hz_zero_phase(&pcm)?;
    let config=ChunkConfig{core_frames:900,..ChunkConfig::default()};
    let default=CostModel::default();
    let attention=CostModel{attention_milli:100,..default};
    for batch_size in [1,2,4] {
        report(&pcm,"quiet900extra1",config,ChunkStrategy::QuietSplice{extra_chunks:1},batch_size,default);
        report(&pcm,"cost-default",config,ChunkStrategy::CostAware(default),batch_size,default);
        report(&pcm,"batch-aware-default",config,ChunkStrategy::BatchAware(default),batch_size,default);
        report(&pcm,"cost-attention",config,ChunkStrategy::CostAware(attention),batch_size,attention);
        report(&pcm,"batch-aware-attention",config,ChunkStrategy::BatchAware(attention),batch_size,attention);
    }
    Ok(())
}
