use std::path::Path;
use burn::tensor::{Device, Tensor, TensorData, activation};
use serde_json::Value;
use crate::checkpoint::{Checkpoint, ModelPaths};
use super::{decode::{DrumOutput, DrumProbabilities, DrumThresholds, decode},
    features::{self, BANDS, DrumFeatures}, net::{DrumNet, OrchestralExpert}};

pub struct DrumTranscriber {
    device: Device,
    kit: DrumNet,
    orchestral: Vec<OrchestralExpert>,
    thresholds: DrumThresholds,
}

fn threshold(json: &Value, key: &str) -> Result<f32, String> {
    let f=json.get(key).and_then(Value::as_f64)
        .ok_or_else(||format!("missing drum threshold {key}"))? as f32;
    if !f.is_finite() || !(0.0..=1.0).contains(&f) {
        return Err(format!("invalid threshold {key}"));
    }
    Ok(f)
}

fn read_probs(t:Tensor<3>)->Result<(usize,Vec<f32>),String>{
    let [batch,frames,_lanes]=t.dims();
    if batch!=1 {return Err("only mono, single-song drum batches are supported".into());}
    Ok((frames,t.into_data().try_into_vec::<f32>()
        .map_err(|e|format!("drum inference readback: {e}"))?))
}

impl DrumTranscriber {
    pub fn from_directory(root: impl AsRef<Path>, device: &Device) -> Result<Self, String> {
        let paths=ModelPaths::in_directory(root,"drums")?;
        Self::from_paths(&paths, device)
    }

    pub fn from_paths(paths: &ModelPaths, device: &Device) -> Result<Self, String> {
        let config=paths.load_config()?;
        let checkpoint=Checkpoint::open(&paths.weights)?;
        Self::load(&config,&checkpoint,device)
    }

    pub fn load(config:&Value, checkpoint:&Checkpoint,device:&Device)->Result<Self,String>{
        if config.get("model_type").and_then(Value::as_str)!=Some("dettato-drums")
            || config.get("sample_rate").and_then(Value::as_u64)!=Some(44100)
            || config.get("feature_fps").and_then(Value::as_u64)!=Some(100)
            || config.get("output_fps").and_then(Value::as_u64)!=Some(50)
        {return Err("unsupported Dettato drums configuration".into());}
        let kit_cfg=config.get("drum_kit").ok_or("missing drum kit config")?;
        let orch_cfg=config.get("orchestral").ok_or("missing orchestral config")?;
        let count=orch_cfg.get("n_models").and_then(Value::as_u64)
            .ok_or("orchestra n_models missing")? as usize;
        if count==0||count>16{return Err("invalid number of orchestra models".into());}
        let adapter=orch_cfg.get("adapter").ok_or("orchestra adapter missing")?;
        let settings=config.get("thresholds").ok_or("missing drum thresholds")?;
        let thresholds=DrumThresholds {
            kit:threshold(settings,"drum_kit")?,
            orchestral:threshold(settings,"orchestral_drum")?,
            timpani:threshold(settings,"timpani")?,
            router:threshold(settings,"router")?,
        };
        let kit=DrumNet::load(checkpoint,"drum_kit",device,kit_cfg)?;
        let orchestral=(0..count).map(|i|{
            OrchestralExpert::load(checkpoint,&format!("orchestral.{i}"),device,kit_cfg,adapter)
        }).collect::<Result<Vec<_>,_>>()?;
        Ok(Self{device:device.clone(),kit,orchestral,thresholds})
    }

    pub fn probabilities(&self,features:&DrumFeatures)->Result<DrumProbabilities,String>{
        let n=features.frames/2*2;
        if n<2||features.data.len()!=features.frames*BANDS {
            return Err("drum features have invalid dimensions".into());
        }
        let x=Tensor::<3>::from_data(TensorData::new(features.data[..n*BANDS].to_vec(),[1,n,BANDS]),&self.device);
        let (_,kit_on,kit_vel)=self.kit.forward_mel(x.clone().narrow(2,0,160))?;
        let (kit_frames,kit)=read_probs(activation::sigmoid(kit_on))?;
        let (_,kit_velocity)=read_probs(activation::sigmoid(kit_vel))?;
        let mut averaged_drum=Vec::<f32>::new();
        let mut averaged_timpani=Vec::<f32>::new();
        let mut averaged_velocity=Vec::<f32>::new();
        let mut minimum_frames=usize::MAX;
        let mut router_sum=0.0;
        for (i,expert) in self.orchestral.iter().enumerate(){
            let output=expert.forward(x.clone())?;
            let (frames,drum)=read_probs(activation::sigmoid(output.drum_logits))?;
            let (_,timpani)=read_probs(activation::sigmoid(output.timpani_logits))?;
            let (_,velocity)=read_probs(activation::sigmoid(output.velocity_logits))?;
            let (_,router)=read_probs(activation::sigmoid(output.router_logits))?;
            router_sum+=router[0];
            minimum_frames=minimum_frames.min(frames);
            if i==0 {
                averaged_drum=drum;
                averaged_timpani=timpani;
                averaged_velocity=velocity;
            } else {
                for (a,b) in averaged_drum.iter_mut().zip(drum){*a+=b;}
                for (a,b) in averaged_timpani.iter_mut().zip(timpani){*a+=b;}
                for (a,b) in averaged_velocity.iter_mut().zip(velocity){*a+=b;}
            }
        }
        let frames=kit_frames.min(minimum_frames);
        if frames==0{return Err("drum models returned no frames".into());}
        averaged_drum.truncate(frames*61);
        averaged_timpani.truncate(frames*13);
        averaged_velocity.truncate(frames*61);
        let reciprocal=1.0/self.orchestral.len() as f32;
        for v in &mut averaged_drum {*v*=reciprocal;}
        for v in &mut averaged_timpani {*v*=reciprocal;}
        for v in &mut averaged_velocity {*v*=reciprocal;}
        Ok(DrumProbabilities{
            router:router_sum*reciprocal,frames,
            kit:kit[..frames*61].to_vec(),kit_velocity:kit_velocity[..frames*61].to_vec(),
            orchestral:averaged_drum,timpani:averaged_timpani,orchestral_velocity:averaged_velocity,
        })
    }

    /// Returns `None` for source shorter than 100 ms (upstream behavior).
    pub fn transcribe(&self,audio44k:&[f32])->Result<Option<DrumOutput>,String>{
        if audio44k.len()<4410{return Ok(None);}
        let feats=features::extract(audio44k)?;
        let prob=self.probabilities(&feats)?;
        Ok(Some(decode(&prob,self.thresholds)?))
    }
}
