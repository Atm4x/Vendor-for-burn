use crate::{checkpoint::Checkpoint, transcriber::model::{BatchNorm, Conv2d}};
use burn::tensor::{Device, Tensor, TensorData, activation, module, ops::{MaxPoolOptions, ConvOptions}};
use serde_json::Value;

use super::{DRUM_KEYS, features::{BANDS, CQ_BANDS, MEL_BANDS}};

#[derive(Clone)]
struct Dense {
    w: Tensor<2>,
    b: Tensor<1>,
}

impl Dense {
    fn load(ckpt: &Checkpoint, key: &str, device: &Device) -> Result<Self, String> {
        let w = ckpt.tensor::<2>(&format!("{key}.weight"), device)?.transpose();
        let b = ckpt.tensor::<1>(&format!("{key}.bias"), device)?;
        if w.dims()[1] != b.dims()[0] { return Err(format!("{key}: bad dense shape")); }
        Ok(Self { w, b })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [i,o] = self.w.dims();
        x.matmul(self.w.clone().reshape([1,i,o])) + self.b.clone().reshape([1,1,o])
    }
}

struct Norm {
    w: Tensor<1>,
    b: Tensor<1>,
}
impl Norm {
    fn load(ckpt: &Checkpoint, key: &str, device: &Device) -> Result<Self, String> {
        let w = ckpt.tensor::<1>(&format!("{key}.weight"), device)?;
        let b = ckpt.tensor::<1>(&format!("{key}.bias"), device)?;
        if w.dims() != b.dims() { return Err(format!("{key}: bad LayerNorm shapes")); }
        Ok(Self { w, b })
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let d = self.w.dims()[0];
        let centered = x.clone() - x.mean_dim(2);
        centered.clone() / (centered.powf_scalar(2.0).mean_dim(2) + 1e-5).sqrt()
            * self.w.clone().reshape([1,1,d]) + self.b.clone().reshape([1,1,d])
    }
}

struct GroupNorm2d {
    w: Tensor<1>,
    b: Tensor<1>,
    groups: usize,
}
impl GroupNorm2d {
    fn load(ckpt: &Checkpoint, key: &str, device: &Device, groups: usize) -> Result<Self,String> {
        let w=ckpt.tensor::<1>(&format!("{key}.weight"),device)?;
        let b=ckpt.tensor::<1>(&format!("{key}.bias"),device)?;
        if groups==0 || w.dims()[0]%groups!=0 || b.dims()!=w.dims() {
            return Err(format!("{key}: invalid group normalization"));
        }
        Ok(Self{w,b,groups})
    }
    fn forward(&self,x:Tensor<4>)->Tensor<4> {
        let [batch,c,t,f]=x.dims();
        let grouped=x.reshape([batch,self.groups,c/self.groups,t*f]);
        let mean=grouped.clone().mean_dim(2).mean_dim(3);
        let centered=grouped-mean;
        let variance=centered.clone().powf_scalar(2.0).mean_dim(2).mean_dim(3);
        let norm=(centered/(variance+1e-5).sqrt()).reshape([batch,c,t,f]);
        norm*self.w.clone().reshape([1,c,1,1])+self.b.clone().reshape([1,c,1,1])
    }
}

#[derive(Clone,Copy)]
enum Activation{Relu,Silu,Gelu}
impl Activation {
    fn from_config(config:&Value)->Result<Self,String> {
        match config.get("act").and_then(Value::as_str).unwrap_or("relu") {
            "relu"=>Ok(Self::Relu),"silu"=>Ok(Self::Silu),"gelu"=>Ok(Self::Gelu),
            other=>Err(format!("unsupported DrumNet activation: {other}")),
        }
    }
    fn apply(&self,x:Tensor<4>)->Tensor<4> {
        match self {Self::Relu=>activation::relu(x),Self::Silu=>activation::silu(x),Self::Gelu=>activation::gelu(x)}
    }
}


struct Conv1As2 {
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    dilation: usize,
    pad: usize,
    groups: usize,
}
impl Conv1As2 {
    fn load(ckpt:&Checkpoint,key:&str,device:&Device,dilation:usize,pad:usize,groups:usize)->Result<Self,String>{
        let raw=ckpt.tensor::<3>(&format!("{key}.weight"),device)?;
        let [out,cin,k]=raw.dims();
        if out%groups!=0 || cin==0 || k==0 {return Err(format!("{key}: invalid Conv1D"));}
        let bias=if ckpt.has(&format!("{key}.bias")){
            Some(ckpt.tensor::<1>(&format!("{key}.bias"),device)?)
        }else{None};
        Ok(Self{weight:raw.reshape([out,cin,k,1]),bias,dilation,pad,groups})
    }
    fn forward(&self,x:Tensor<4>)->Tensor<4>{
        module::conv2d(x,self.weight.clone(),self.bias.clone(),
            ConvOptions::new([1,1],[self.pad,0],[self.dilation,1],self.groups))
    }
}

struct GruDirection{
    wi: Tensor<2>,
    wh: Tensor<2>,
    bi: Tensor<1>,
    bh: Tensor<1>,
    recurrent_weights: Vec<f32>,
    recurrent_bias: Vec<f32>,
}
impl GruDirection {
    fn load(ckpt:&Checkpoint,key:&str,layer:usize,reverse:bool,device:&Device)->Result<Self,String>{
        let suffix=if reverse{"_reverse"}else{""};
        let prefix=format!("{key}.");
        let wi=ckpt.tensor::<2>(&format!("{prefix}weight_ih_l{layer}{suffix}"),device)?.transpose();
        let (shape, weights) = ckpt.values(&format!("{prefix}weight_hh_l{layer}{suffix}"))?;
        let [gates, hidden]: [usize; 2] = shape.try_into()
            .map_err(|_| format!("{key}: expected a matrix of recurrent weights"))?;
        let bi=ckpt.tensor::<1>(&format!("{prefix}bias_ih_l{layer}{suffix}"),device)?;
        let (bias_shape, recurrent_bias) = ckpt.values(&format!("{prefix}bias_hh_l{layer}{suffix}"))?;
        if hidden == 0 || gates != 3 * hidden || wi.dims()[1] != gates || bi.dims() != [gates] || bias_shape != [gates] {
            return Err(format!("{key}: incompatible GRU gates at layer {layer}"));
        }
        let mut recurrent_weights = vec![0.0; weights.len()];
        for (gate, row) in weights.chunks_exact(hidden).enumerate() {
            for (unit, &weight) in row.iter().enumerate() {
                recurrent_weights[unit * gates + gate] = weight;
            }
        }
        let wh = Tensor::from_data(TensorData::new(recurrent_weights.clone(), [hidden, gates]), device);
        let bh = Tensor::from_data(TensorData::new(recurrent_bias.clone(), [gates]), device);
        Ok(Self { wi, wh, bi, bh, recurrent_weights, recurrent_bias })
    }
    fn forward(&self, x: Tensor<3>, reverse: bool) -> Result<Tensor<3>, String> {
        let [batch, time, input_dim] = x.dims();
        let hidden=self.wh.dims()[0];
        let device=x.device();
        let input_gates = x.matmul(self.wi.clone().reshape([1, input_dim, 3 * hidden]))
            + self.bi.clone().reshape([1, 1, 3 * hidden]);
        // For a single song, the small recurrent matrix-vector product is
        // cheaper to scan on the host than to dispatch per timestep.
        if batch == 1 && time >= 64 {
            let input = input_gates.cast(burn::tensor::DType::F32).into_data()
                .try_into_vec::<f32>().map_err(|error| format!("Drum GRU gate readback: {error}"))?;
            let output = recurrent_scan(&input, &self.recurrent_weights, &self.recurrent_bias, reverse);
            return Ok(Tensor::from_data(TensorData::new(output, [1, time, hidden]), &device));
        }
        let mut state=Tensor::<2>::zeros([batch,hidden],&device);
        let mut outputs:Vec<Option<Tensor<3>>>=vec![None;time];
        for step in 0..time {
            let t=if reverse{time-1-step}else{step};
            let xg = input_gates.clone().narrow(1, t, 1).reshape([batch, 3 * hidden]);
            let hg=state.clone().matmul(self.wh.clone())+self.bh.clone().reshape([1,3*hidden]);
            let r=activation::sigmoid(xg.clone().narrow(1,0,hidden)+hg.clone().narrow(1,0,hidden));
            let z=activation::sigmoid(xg.clone().narrow(1,hidden,hidden)+hg.clone().narrow(1,hidden,hidden));
            let n=(xg.narrow(1,2*hidden,hidden)+r*hg.narrow(1,2*hidden,hidden)).tanh();
            state=(Tensor::<2>::ones([batch,hidden],&device)-z.clone())*n+z*state;
            outputs[t]=Some(state.clone().reshape([batch,1,hidden]));
        }
        Ok(Tensor::cat(outputs.into_iter().map(Option::unwrap).collect(),1))
    }
}

fn recurrent_scan(input: &[f32], weights: &[f32], bias: &[f32], reverse: bool) -> Vec<f32> {
    let gate_count = bias.len();
    let hidden = gate_count / 3;
    let steps = input.len() / gate_count;
    let mut output = vec![0.0; steps * hidden];
    let mut state = vec![0.0; hidden];
    let mut gates = vec![0.0; gate_count];
    let sigmoid = |value: f32| 1.0 / (1.0 + (-value).exp());

    for step in 0..steps {
        let time = if reverse { steps - 1 - step } else { step };
        let input = &input[time * gate_count..(time + 1) * gate_count];
        gates.fill(0.0);
        for (&unit, weights) in state.iter().zip(weights.chunks_exact(gate_count)) {
            for (gate, &weight) in gates.iter_mut().zip(weights) {
                *gate += unit * weight;
            }
        }
        for (gate, &bias) in gates.iter_mut().zip(bias) {
            *gate += bias;
        }
        for (unit, state) in state.iter_mut().enumerate() {
            let reset = sigmoid(input[unit] + gates[unit]);
            let update = sigmoid(input[hidden + unit] + gates[hidden + unit]);
            let candidate = (input[2 * hidden + unit] + reset * gates[2 * hidden + unit]).tanh();
            *state = (1.0 - update) * candidate + update * *state;
        }
        output[time * hidden..(time + 1) * hidden].copy_from_slice(&state);
    }
    output
}

struct BiGru {
    directions:Vec<(GruDirection,GruDirection)>,
}
impl BiGru {
    fn load(ckpt:&Checkpoint,key:&str,layers:usize,device:&Device)->Result<Self,String>{
        let mut directions=Vec::with_capacity(layers);
        for l in 0..layers {
            directions.push((GruDirection::load(ckpt,key,l,false,device)?,
                GruDirection::load(ckpt,key,l,true,device)?));
        }
        Ok(Self{directions})
    }
    fn forward(&self,mut x:Tensor<3>)->Result<Tensor<3>,String>{
        for (fwd,bwd) in &self.directions {
            let a=fwd.forward(x.clone(),false)?;
            let b=bwd.forward(x,true)?;
            x=Tensor::cat(vec![a,b],2);
        }
        Ok(x)
    }
}

enum StemStep {
    Conv{conv:Conv2d,norm:BatchNorm,act:Activation},
    Residual{dw:Conv2d,n1:BatchNorm,pw:Conv2d,n2:BatchNorm,act:Activation},
    Pool([usize;2]),
}
struct DrumStem {layers:Vec<StemStep>,channels:usize,time_divisor:usize,frequency_divisor:usize}
impl DrumStem {
    fn load(ckpt:&Checkpoint,prefix:&str,device:&Device,spec:&str,act:Activation)->Result<Self,String>{
        let mut index=0usize;
        let mut channels=1usize;
        let mut time_divisor=1usize;
        let mut frequency_divisor=1usize;
        let mut layers=Vec::new();
        for token in spec.split(',') {
            let token=token.trim();
            let (kind,arg)=token.split_at(1);
            let layer=match kind {
                "c"=>{
                    let next=arg.parse::<usize>().map_err(|e|e.to_string())?;
                    if next==0 {return Err("zero stem channels".into());}
                    let conv=Conv2d::load(ckpt,&format!("{prefix}.{index}"),device,[1,1],[1,1],[1,1],1)?;
                    let norm=BatchNorm::load(ckpt,&format!("{prefix}.{}",index+1),device)?;
                    index+=3;channels=next;
                    StemStep::Conv{conv,norm,act}
                }
                "r"=>{
                    let key=format!("{prefix}.{index}");
                    index+=1;
                    StemStep::Residual{
                        dw:Conv2d::load(ckpt,&format!("{key}.dw"),device,[1,1],[1,1],[1,1],channels)?,
                        n1:BatchNorm::load(ckpt,&format!("{key}.bn1"),device)?,
                        pw:Conv2d::load(ckpt,&format!("{key}.pw1"),device,[1,1],[0,0],[1,1],1)?,
                        n2:BatchNorm::load(ckpt,&format!("{key}.bn2"),device)?,act,
                    }
                }
                "p"=>{
                    if arg.len()<2{return Err(format!("invalid pool specification: {token}"));}
                    let t=arg[..1].parse::<usize>().map_err(|e|e.to_string())?;
                    let f=arg[1..].parse::<usize>().map_err(|e|e.to_string())?;
                    if t==0||f==0 {return Err("zero pool factor".into());}
                    index+=1;time_divisor*=t;frequency_divisor*=f;
                    StemStep::Pool([t,f])
                }
                _=>return Err(format!("unknown DrumNet stem step: {token}")),
            };
            layers.push(layer);
        }
        if time_divisor!=2 || frequency_divisor!=8 {
            return Err(format!("DrumNet stem stride mismatch: {time_divisor}x{frequency_divisor}"));
        }
        Ok(Self{layers,channels,time_divisor,frequency_divisor})
    }
    fn forward(&self,mut x:Tensor<4>)->Tensor<4>{
        for layer in &self.layers {
            x=match layer {
                StemStep::Conv{conv,norm,act}=>act.apply(norm.forward(conv.forward(x))),
                StemStep::Residual{dw,n1,pw,n2,act}=>{
                    let h=n2.forward(pw.forward(act.apply(n1.forward(dw.forward(x.clone())))));
                    x+h
                }
                StemStep::Pool(shape)=>module::max_pool2d(x,MaxPoolOptions::new(*shape)),
            };
        }
        x
    }
}

struct BandCollapse {rnn:BiGru,proj:Dense}
impl BandCollapse {
    fn load(ckpt:&Checkpoint,prefix:&str,device:&Device)->Result<Self,String>{
        Ok(Self{rnn:BiGru::load(ckpt,&format!("{prefix}.rnn"),1,device)?,
            proj:Dense::load(ckpt,&format!("{prefix}.proj"),device)?})
    }
    fn forward(&self,x:Tensor<4>)->Result<Tensor<3>,String>{
        let [batch,channels,time,freq]=x.dims();
        let z=x.permute([0,2,3,1]).reshape([batch*time,freq,channels]);
        let r=self.rnn.forward(z)?;
        let hidden=r.dims()[2]/2;
        let final_forward=r.clone().narrow(1,freq-1,1).narrow(2,0,hidden);
        let final_backward=r.clone().narrow(1,0,1).narrow(2,hidden,hidden);
        let mean=r.mean_dim(1);
        Ok(self.proj.forward(Tensor::cat(vec![final_forward,final_backward,mean],2))
            .reshape([batch,time,self.proj.b.dims()[0]]))
    }
}

enum Temporal {
    Convolution{dw:Conv1As2,bn:BatchNorm,pw:Conv1As2},
    Gru{norm:Norm,rnn:BiGru,output:Dense},
    Attention{norm:Norm,qkv:Dense,output:Dense,heads:usize},
}
fn content_attention(qkv:Tensor<3>,heads:usize)->Result<Tensor<3>,String>{
    let [b,t,three_d]=qkv.dims();
    let dim=three_d/3;
    if b == 0 || t == 0 || heads == 0 || dim == 0 || three_d!=3*dim || dim%heads!=0{return Err("attention dimension mismatch".into());}
    let head_dim=dim/heads;
    let proj=qkv.reshape([b,t,3,heads,head_dim]);
    let get=|idx:usize|proj.clone().narrow(2,idx,1).reshape([b,t,heads,head_dim]).permute([0,2,1,3]);
    let q=get(0);let k=get(1);let v=get(2);
    let key = k.transpose();
    let scale = (head_dim as f32).powf(-0.5);
    let attend = |query: Tensor<4>| {
        activation::softmax(query.matmul(key.clone()) * scale, 3).matmul(v.clone())
    };
    // Each query tile still attends to every key. Bound the score buffer for
    // full recordings and keep per-row softmax independent of tile size.
    let result = if t <= 1024 {
        attend(q)
    } else {
        let tile = (8 * 1024 * 1024 / b.saturating_mul(heads).saturating_mul(t))
            .clamp(1, 256);
        let chunks = (0..t).step_by(tile)
            .map(|start| attend(q.clone().narrow(2, start, (t - start).min(tile))))
            .collect();
        Tensor::cat(chunks, 2)
    };
    Ok(result.permute([0,2,1,3]).reshape([b,t,dim]))
}
impl Temporal {
    fn load(ckpt:&Checkpoint,key:&str,device:&Device,spec:&Value)->Result<Self,String>{
        match spec.get("t").and_then(Value::as_str).ok_or("missing temporal block type")? {
            "tb"=>{
                let k=spec.get("k").and_then(Value::as_u64).unwrap_or(15) as usize;
                let d=spec.get("d").and_then(Value::as_u64).unwrap_or(1) as usize;
                let w=ckpt.tensor::<3>(&format!("{key}.dw.weight"),device)?.dims()[0];
                Ok(Self::Convolution{
                    dw:Conv1As2::load(ckpt,&format!("{key}.dw"),device,d,d*(k/2),w)?,
                    bn:BatchNorm::load(ckpt,&format!("{key}.bn"),device)?,
                    pw:Conv1As2::load(ckpt,&format!("{key}.pw"),device,1,0,1)?,
                })
            }
            "gru"=>Ok(Self::Gru{
                norm:Norm::load(ckpt,&format!("{key}.norm"),device)?,
                rnn:BiGru::load(ckpt,&format!("{key}.rnn"),1,device)?,
                output:Dense::load(ckpt,&format!("{key}.out"),device)?,
            }),
            "attn"=>Ok(Self::Attention{
                norm:Norm::load(ckpt,&format!("{key}.n"),device)?,
                qkv:Dense::load(ckpt,&format!("{key}.qkv"),device)?,
                output:Dense::load(ckpt,&format!("{key}.o"),device)?,
                heads:spec.get("heads").and_then(Value::as_u64).unwrap_or(2) as usize,
            }),
            other=>Err(format!("unsupported temporal block: {other}")),
        }
    }
    fn forward(&self,x:Tensor<3>)->Result<Tensor<3>,String>{
        Ok(match self {
            Self::Convolution{dw,bn,pw}=>{
                let [b,t,w]=x.dims();
                let z=x.clone().permute([0,2,1]).reshape([b,w,t,1]);
                let z=pw.forward(activation::silu(bn.forward(dw.forward(z))));
                x+z.reshape([b,w,t]).permute([0,2,1])
            }
            Self::Gru{norm,rnn,output}=>x.clone()+output.forward(rnn.forward(norm.forward(x))?),
            Self::Attention{norm,qkv,output,heads}=>{
                let z=qkv.forward(norm.forward(x.clone()));
                x+output.forward(content_attention(z,*heads)?)
            }
        })
    }
}

struct FamilyHeads {
    rest:Dense,
    towers:Vec<BiGru>,
    heads:Vec<Dense>,
    scatter:Tensor<3>,
    keep_rest:Tensor<3>,
}
impl FamilyHeads {
    fn load(ckpt:&Checkpoint,prefix:&str,device:&Device,config:&Value)->Result<Self,String>{
        let groups=config.get("towers").and_then(Value::as_array).ok_or("missing drum family towers")?;
        let mut towers=Vec::new();let mut heads=Vec::new();let mut indices=Vec::new();
        for (i,group) in groups.iter().enumerate(){
            towers.push(BiGru::load(ckpt,&format!("{prefix}.towers.{i}"),1,device)?);
            heads.push(Dense::load(ckpt,&format!("{prefix}.heads.{i}"),device)?);
            for key in group.as_array().ok_or("invalid drum tower group")? {
                let key=key.as_u64().ok_or("invalid drum key")?;
                if !(27..=87).contains(&key){return Err("drum key out of range".into());}
                indices.push(key as usize-27);
            }
        }
        if indices.len()!=heads.iter().map(|h|h.b.dims()[0]).sum::<usize>() {
            return Err("drum towers and projection dimensions differ".into());
        }
        let mut scatter=vec![0f32;indices.len()*DRUM_KEYS];
        let mut keep=vec![1f32;DRUM_KEYS];
        for (row,&index) in indices.iter().enumerate(){
            if keep[index]==0.0 {return Err("duplicate drum tower key".into());}
            scatter[row*DRUM_KEYS+index]=1.0;
            keep[index]=0.0;
        }
        Ok(Self{
            rest:Dense::load(ckpt,&format!("{prefix}.rest"),device)?,towers,heads,
            scatter:Tensor::from_data(TensorData::new(scatter,[1,indices.len(),DRUM_KEYS]),device),
            keep_rest:Tensor::from_data(TensorData::new(keep,[1,1,DRUM_KEYS]),device),
        })
    }
    fn forward(&self,h:Tensor<3>)->Result<Tensor<3>,String>{
        let rest=self.rest.forward(h.clone())*self.keep_rest.clone();
        let mut parts=Vec::new();
        for (rnn,head) in self.towers.iter().zip(&self.heads){
            let z=rnn.forward(h.clone())?;
            parts.push(head.forward(Tensor::cat(vec![z,h.clone()],2)));
        }
        Ok(rest + Tensor::cat(parts, 2).matmul(self.scatter.clone()))
    }
}

pub struct DrumNet {
    stem:DrumStem,collapse:BandCollapse,temporal:Vec<Temporal>,families:FamilyHeads,
    vel1:Dense,vel2:Dense,device:Device,
}
impl DrumNet {
    pub fn load(ckpt:&Checkpoint,prefix:&str,device:&Device,config:&Value)->Result<Self,String>{
        let spec=config.get("stem").and_then(Value::as_str).ok_or("missing drum stem")?;
        let stem=DrumStem::load(ckpt,&format!("{prefix}.stem.net"),device,spec,Activation::from_config(config)?)?;
        let mut temporal=Vec::new();
        for spec in config.get("temporal").and_then(Value::as_array).ok_or("missing temporal specification")? {
            let count=spec.get("n").and_then(Value::as_u64).unwrap_or(1);
            for _ in 0..count {
                temporal.push(Temporal::load(ckpt,&format!("{prefix}.temporal.{}",temporal.len()),device,spec)?);
            }
        }
        Ok(Self{
            stem,collapse:BandCollapse::load(ckpt,&format!("{prefix}.collapse"),device)?,temporal,
            families:FamilyHeads::load(ckpt,&format!("{prefix}.on"),device,config)?,
            vel1:Dense::load(ckpt,&format!("{prefix}.vel.0"),device)?,
            vel2:Dense::load(ckpt,&format!("{prefix}.vel.1"),device)?,device:device.clone(),
        })
    }
    fn embed(&self,mel:Tensor<3>)->Result<Tensor<3>,String>{
        let [batch,t,band]=mel.dims();
        assert_eq!(band,MEL_BANDS);
        let z=(mel+5.0)/3.0;
        self.collapse.forward(self.stem.forward(z.reshape([batch,1,t,band])))
    }
    fn trunk(&self,emb:Tensor<3>)->Result<(Tensor<3>,Tensor<3>,Tensor<3>),String>{
        let mut h=emb;
        for block in &self.temporal {h=block.forward(h)?;}
        let on=self.families.forward(h.clone())?;
        let vel=self.vel2.forward(self.vel1.forward(h.clone()));
        Ok((h,on,vel))
    }
    fn embed_long(&self,x:Tensor<3>)->Result<Tensor<3>,String>{
        let [batch,t,_]=x.dims();
        if t<=9000{return self.embed(x);}
        let mut chunks=Vec::new();
        let mut start=0;
        while start<t {
            let a=start.saturating_sub(200);
            let b=(start+8000+200).min(t);
            let segment=self.embed(x.clone().narrow(1,a,b-a))?;
            let lo=(start-a)/2;
            let n=((start+8000).min(t)-start)/2;
            chunks.push(segment.narrow(1,lo,n));
            start+=8000;
        }
        let output=Tensor::cat(chunks,1);
        assert_eq!(output.dims()[0],batch);
        Ok(output)
    }
    pub fn forward_mel(&self,mel:Tensor<3>)->Result<(Tensor<3>,Tensor<3>,Tensor<3>),String>{
        if mel.dims()[1]<2 || mel.dims()[1]%2!=0{return Err("DrumNet expects even positive frames".into());}
        self.trunk(self.embed_long(mel)?)
    }
    pub fn device(&self)->&Device{&self.device}
}

struct CqBranch {
    c1:Conv2d,c2:Conv2d,c3:Conv2d,
    n1:GroupNorm2d,n2:GroupNorm2d,n3:GroupNorm2d,
    proj:Dense,
}
impl CqBranch {
    fn load(ckpt:&Checkpoint,prefix:&str,device:&Device)->Result<Self,String>{
        Ok(Self{
            c1:Conv2d::load(ckpt,&format!("{prefix}.c1"),device,[1,1],[2,3],[1,1],1)?,
            c2:Conv2d::load(ckpt,&format!("{prefix}.c2"),device,[2,1],[1,2],[1,1],1)?,
            c3:Conv2d::load(ckpt,&format!("{prefix}.c3"),device,[1,1],[1,2],[1,1],1)?,
            n1:GroupNorm2d::load(ckpt,&format!("{prefix}.n1"),device,4)?,
            n2:GroupNorm2d::load(ckpt,&format!("{prefix}.n2"),device,4)?,
            n3:GroupNorm2d::load(ckpt,&format!("{prefix}.n3"),device,4)?,
            proj:Dense::load(ckpt,&format!("{prefix}.proj"),device)?,
        })
    }
    fn forward(&self,x:Tensor<3>)->Tensor<3>{
        let [b,t,f]=x.dims();
        assert_eq!(f,CQ_BANDS);
        let centred=(x.clone()-x.mean_dim(1).mean_dim(2))/4.0;
        let floor=centred.clone().min_dim(1).min_dim(2);
        let shift=|k:usize|{
            Tensor::cat(vec![centred.clone().narrow(2,k,f-k),
                floor.clone().expand([b,t,k])],2)
        };
        let input=Tensor::cat(vec![
            centred.clone().reshape([b,1,t,f]),
            shift(14).reshape([b,1,t,f]),
            shift(24).reshape([b,1,t,f]),
        ],1);
        let z=activation::silu(self.n1.forward(self.c1.forward(input)));
        let z=activation::silu(self.n2.forward(self.c2.forward(z)));
        let z=activation::silu(self.n3.forward(self.c3.forward(z.clone())))+z;
        let [_,channels,frames,bands]=z.dims();
        activation::silu(self.proj.forward(z.permute([0,2,1,3]).reshape([b,frames,channels*bands])))
    }
}

pub struct OrchestraOutput {
    pub drum_logits:Tensor<3>,
    pub timpani_logits:Tensor<3>,
    pub velocity_logits:Tensor<3>,
    pub router_logits:Tensor<3>,
}

pub struct OrchestralExpert {
    base:DrumNet,cq:CqBranch,inp:Dense,rnn:BiGru,
    qkv:Dense,output:Dense,norm:Norm,
    delta:Dense,timpani:Dense,gate1:Dense,gate2:Dense,heads:usize,
}
impl OrchestralExpert {
    pub fn load(ckpt:&Checkpoint,prefix:&str,device:&Device,base_cfg:&Value,adapter:&Value)->Result<Self,String>{
        let layers=adapter.get("layers").and_then(Value::as_u64).unwrap_or(2) as usize;
        let heads=adapter.get("heads").and_then(Value::as_u64).unwrap_or(4) as usize;
        if layers==0||heads==0{return Err("invalid orchestra adapter settings".into());}
        Ok(Self{
            base:DrumNet::load(ckpt,&format!("{prefix}.base"),device,base_cfg)?,
            cq:CqBranch::load(ckpt,&format!("{prefix}.cq"),device)?,
            inp:Dense::load(ckpt,&format!("{prefix}.inp"),device)?,
            rnn:BiGru::load(ckpt,&format!("{prefix}.rnn"),layers,device)?,
            qkv:Dense::load(ckpt,&format!("{prefix}.qkv"),device)?,
            output:Dense::load(ckpt,&format!("{prefix}.ao"),device)?,
            norm:Norm::load(ckpt,&format!("{prefix}.ln"),device)?,
            delta:Dense::load(ckpt,&format!("{prefix}.delta"),device)?,
            timpani:Dense::load(ckpt,&format!("{prefix}.timp"),device)?,
            gate1:Dense::load(ckpt,&format!("{prefix}.gate.0"),device)?,
            gate2:Dense::load(ckpt,&format!("{prefix}.gate.2"),device)?,
            heads,
        })
    }
    pub fn forward(&self,x:Tensor<3>)->Result<OrchestraOutput,String>{
        let [b,t,f]=x.dims();
        if t<2||f!=BANDS{return Err("orchestra features must have 280 bands".into());}
        let t=t/2*2;
        let x=x.narrow(1,0,t);
        let mel=x.clone().narrow(2,0,MEL_BANDS);
        let (base_hidden,base_onset,base_velocity)=self.base.forward_mel(mel)?;
        let cq=self.cq.forward(x.narrow(2,MEL_BANDS,CQ_BANDS));
        let frames=base_hidden.dims()[1].min(cq.dims()[1]);
        let h0=base_hidden.narrow(1,0,frames);
        let on=base_onset.narrow(1,0,frames);
        let vel=base_velocity.narrow(1,0,frames);
        let cq=cq.narrow(1,0,frames);
        let input=activation::silu(self.inp.forward(Tensor::cat(vec![h0,cq,
            activation::sigmoid(on.clone())],2)));
        let r=self.rnn.forward(input)?;
        let attention=content_attention(self.qkv.forward(r.clone()),self.heads)?;
        let norm=self.norm.forward(r+self.output.forward(attention));
        let pooled=norm.clone().mean_dim(1);
        let router=self.gate2.forward(activation::silu(self.gate1.forward(pooled)));
        assert_eq!(router.dims(),[b,1,1]);
        Ok(OrchestraOutput{
            drum_logits:on+self.delta.forward(norm.clone()),
            timpani_logits:self.timpani.forward(norm),
            velocity_logits:vel,router_logits:router,
        })
    }
}

#[cfg(test)]
mod recurrent_tests {
    use super::recurrent_scan;

    #[test]
    fn recurrent_scan_preserves_direction_and_gate_order() {
        let input = [0.2, -0.1, 0.7, -0.3, 0.6, -0.5, 0.4, 0.1, 0.9];
        let weights = [-0.2, 0.3, 0.4];
        let bias = [0.01, -0.02, 0.03];
        for reverse in [false, true] {
            let result = recurrent_scan(&input, &weights, &bias, reverse);
            let mut state = 0.0_f64;
            for step in 0..3 {
                let time = if reverse { 2 - step } else { step };
                let offset = time * 3;
                let r = 1.0 / (1.0 + (-(f64::from(input[offset]) + state * f64::from(weights[0]) + f64::from(bias[0]))).exp());
                let z = 1.0 / (1.0 + (-(f64::from(input[offset + 1]) + state * f64::from(weights[1]) + f64::from(bias[1]))).exp());
                let candidate = (f64::from(input[offset + 2]) + r * (state * f64::from(weights[2]) + f64::from(bias[2]))).tanh();
                state = candidate + z * (state - candidate);
                assert!((f64::from(result[time]) - state).abs() < 1e-6);
            }
        }
    }
}
