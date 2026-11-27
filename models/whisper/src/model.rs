use crate::weights::{RawTensor, Weights};
use anyhow::{Result, bail};
use burn::tensor::{
    Bool, Device, Int, Tensor, TensorData, activation,
    module::{attention, conv1d},
    ops::{AttentionModuleOptions, ConvOptions},
};

const LN_EPS: f32 = 1e-5;
const HEAD_DIM: usize = 64;

fn t1(r: RawTensor, dev: &Device) -> Tensor<1> {
    let n = r.shape[0];
    Tensor::from_data(TensorData::new(r.data, [n]), dev)
}
fn t2(r: RawTensor, dev: &Device) -> Tensor<2> {
    let s = [r.shape[0], r.shape[1]];
    Tensor::from_data(TensorData::new(r.data, s), dev)
}
fn t3(r: RawTensor, dev: &Device) -> Tensor<3> {
    let s = [r.shape[0], r.shape[1], r.shape[2]];
    Tensor::from_data(TensorData::new(r.data, s), dev)
}

fn transpose_raw(r: RawTensor) -> RawTensor {
    let (rows, cols) = (r.shape[0], r.shape[1]);
    let mut data = vec![0.0f32; r.data.len()];
    for i in 0..rows {
        for j in 0..cols {
            data[j * rows + i] = r.data[i * cols + j];
        }
    }
    RawTensor {
        shape: vec![cols, rows],
        data,
    }
}

struct Linear {
    w: Tensor<2>,
    b: Option<Tensor<1>>,
}

impl Linear {
    fn load(w: &mut Weights, name: &str, bias: bool, dev: &Device) -> Result<Self> {
        let weight = t2(transpose_raw(w.take(&format!("{name}.weight"))?), dev);
        let b = if bias {
            Some(t1(w.take(&format!("{name}.bias"))?, dev))
        } else {
            None
        };
        Ok(Self { w: weight, b })
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let y = x.matmul(self.w.clone());
        match &self.b {
            Some(b) => y + b.clone().unsqueeze::<2>(),
            None => y,
        }
    }
}

struct LayerNorm {
    g: Tensor<1>,
    b: Tensor<1>,
}

impl LayerNorm {
    fn load(w: &mut Weights, name: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            g: t1(w.take(&format!("{name}.weight"))?, dev),
            b: t1(w.take(&format!("{name}.bias"))?, dev),
        })
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let xc = x.clone() - x.mean_dim(1);
        let var = (xc.clone() * xc.clone()).mean_dim(1);
        xc / (var + LN_EPS).sqrt() * self.g.clone().unsqueeze::<2>()
            + self.b.clone().unsqueeze::<2>()
    }
}

pub fn sdpa_manual(q: Tensor<3>, k: Tensor<3>, v: Tensor<3>) -> Tensor<3> {
    let hd = q.dims()[2];
    let scores = q.matmul(k.swap_dims(1, 2)) * (hd as f64).powf(-0.5);
    activation::softmax(scores, 2).matmul(v)
}

pub fn sdpa_fused(q: Tensor<3>, k: Tensor<3>, v: Tensor<3>) -> Tensor<3> {
    let [h, t, hd] = q.dims();
    let options = AttentionModuleOptions {
        scale: Some((hd as f64).powf(-0.5)),
        softcap: None,
        is_causal: false,
    };
    attention(
        q.unsqueeze::<4>(),
        k.unsqueeze::<4>(),
        v.unsqueeze::<4>(),
        None,
        None,
        options,
    )
    .reshape([h, t, hd])
}

fn contiguous3(t: Tensor<3>) -> Tensor<3> {
    let dev = t.device();
    Tensor::from_data(t.into_data(), &dev)
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
}

impl Attention {
    fn load(w: &mut Weights, name: &str, heads: usize, dev: &Device) -> Result<Self> {
        Ok(Self {
            q: Linear::load(w, &format!("{name}.q_proj"), true, dev)?,
            k: Linear::load(w, &format!("{name}.k_proj"), false, dev)?,
            v: Linear::load(w, &format!("{name}.v_proj"), true, dev)?,
            o: Linear::load(w, &format!("{name}.out_proj"), true, dev)?,
            heads,
        })
    }

    fn split(&self, x: Tensor<2>) -> Tensor<3> {
        let [t, d] = x.dims();
        x.reshape([t, self.heads, d / self.heads]).swap_dims(0, 1)
    }

    fn project_kv(&self, x: &Tensor<2>) -> (Tensor<3>, Tensor<3>) {
        (
            self.split(self.k.forward(x.clone())),
            self.split(self.v.forward(x.clone())),
        )
    }

    fn attend(
        &self,
        xq: Tensor<2>,
        k: Tensor<3>,
        v: Tensor<3>,
        mask: Option<Tensor<4, Bool>>,
    ) -> Tensor<2> {
        let [t, d] = xq.dims();
        let hd = d / self.heads;
        let q = self.split(self.q.forward(xq)).unsqueeze::<4>();
        let options = AttentionModuleOptions {
            scale: Some((hd as f64).powf(-0.5)),
            softcap: None,
            is_causal: false,
        };
        let out = attention(
            q,
            k.unsqueeze::<4>(),
            v.unsqueeze::<4>(),
            mask,
            None,
            options,
        );
        let out = out
            .reshape([self.heads, t, hd])
            .swap_dims(0, 1)
            .reshape([t, d]);
        self.o.forward(out)
    }
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn load(w: &mut Weights, prefix: &str, dev: &Device) -> Result<Self> {
        Ok(Self {
            fc1: Linear::load(w, &format!("{prefix}.fc1"), true, dev)?,
            fc2: Linear::load(w, &format!("{prefix}.fc2"), true, dev)?,
        })
    }
    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.fc2.forward(activation::gelu(self.fc1.forward(x)))
    }
}

struct EncoderBlock {
    attn_ln: LayerNorm,
    attn: Attention,
    mlp_ln: LayerNorm,
    mlp: Mlp,
}

struct DecoderBlock {
    attn_ln: LayerNorm,
    attn: Attention,
    cross_ln: LayerNorm,
    cross: Attention,
    mlp_ln: LayerNorm,
    mlp: Mlp,
}

pub struct Whisper {
    device: Device,
    conv1_w: Tensor<3>,
    conv1_b: Tensor<1>,
    conv2_w: Tensor<3>,
    conv2_b: Tensor<1>,
    enc_pos: Tensor<2>,
    enc_blocks: Vec<EncoderBlock>,
    enc_ln: LayerNorm,
    tok_emb: Tensor<2>,
    dec_pos: Tensor<2>,
    dec_blocks: Vec<DecoderBlock>,
    dec_ln: LayerNorm,
    pub n_vocab: usize,
    pub d_model: usize,
    pub n_mels: usize,
    pub n_text_ctx: usize,
}

#[derive(Default, Debug)]
pub struct Prof {
    pub conv: f64,
    pub layer_norm: f64,
    pub kv_proj: f64,
    pub attention: f64,
    pub mlp: f64,
    pub residual: f64,
}

#[derive(Default, Debug)]
pub struct DecProf {
    pub embed: f64,
    pub self_attn: f64,
    pub cross_attn: f64,
    pub mlp: f64,
    pub logits: f64,
}

struct Lap {
    on: bool,
    t: std::time::Instant,
}
impl Lap {
    fn tick(&mut self, acc: &mut f64) {
        if self.on {
            *acc += self.t.elapsed().as_secs_f64() * 1e3;
            self.t = std::time::Instant::now();
        }
    }
}
fn sync2(on: bool, t: &Tensor<2>) {
    if on {
        let _ = t.clone().slice([0..1, 0..1]).into_data();
    }
}
fn sync3(on: bool, t: &Tensor<3>) {
    if on {
        let _ = t.clone().slice([0..1, 0..1, 0..1]).into_data();
    }
}

pub struct DecoderState {
    cross: Vec<(Tensor<3>, Tensor<3>)>,
    selfkv: Vec<Option<(Tensor<3>, Tensor<3>)>>,
    pos: usize,
}

impl DecoderState {
    pub fn position(&self) -> usize {
        self.pos
    }
}

impl Whisper {
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn load(mut w: Weights, device: &Device) -> Result<Self> {
        let dev = device;
        let conv1_raw = w.take("encoder.conv1.weight")?;
        let d_model = conv1_raw.shape[0];
        let n_mels = conv1_raw.shape[1];
        if d_model % HEAD_DIM != 0 {
            bail!("d_model {d_model} is not a multiple of {HEAD_DIM}");
        }
        let heads = d_model / HEAD_DIM;

        let mut n_enc = 0;
        while w.has(&format!("encoder.layers.{n_enc}.fc1.weight")) {
            n_enc += 1;
        }
        let mut n_dec = 0;
        while w.has(&format!("decoder.layers.{n_dec}.fc1.weight")) {
            n_dec += 1;
        }
        if n_enc == 0 || n_dec == 0 {
            bail!("no encoder/decoder layers found in weights");
        }

        let conv1_w = t3(conv1_raw, dev);
        let conv1_b = t1(w.take("encoder.conv1.bias")?, dev);
        let conv2_w = t3(w.take("encoder.conv2.weight")?, dev);
        let conv2_b = t1(w.take("encoder.conv2.bias")?, dev);
        let enc_pos = t2(w.take("encoder.embed_positions.weight")?, dev);

        let mut enc_blocks = Vec::new();
        for i in 0..n_enc {
            let p = format!("encoder.layers.{i}");
            enc_blocks.push(EncoderBlock {
                attn_ln: LayerNorm::load(&mut w, &format!("{p}.self_attn_layer_norm"), dev)?,
                attn: Attention::load(&mut w, &format!("{p}.self_attn"), heads, dev)?,
                mlp_ln: LayerNorm::load(&mut w, &format!("{p}.final_layer_norm"), dev)?,
                mlp: Mlp::load(&mut w, &p, dev)?,
            });
        }
        let enc_ln = LayerNorm::load(&mut w, "encoder.layer_norm", dev)?;

        let tok_raw = w.take("decoder.embed_tokens.weight")?;
        let n_vocab = tok_raw.shape[0];
        let tok_emb = t2(tok_raw, dev);
        let dec_pos_raw = w.take("decoder.embed_positions.weight")?;
        let n_text_ctx = dec_pos_raw.shape[0];
        let dec_pos = t2(dec_pos_raw, dev);

        let mut dec_blocks = Vec::new();
        for i in 0..n_dec {
            let p = format!("decoder.layers.{i}");
            dec_blocks.push(DecoderBlock {
                attn_ln: LayerNorm::load(&mut w, &format!("{p}.self_attn_layer_norm"), dev)?,
                attn: Attention::load(&mut w, &format!("{p}.self_attn"), heads, dev)?,
                cross_ln: LayerNorm::load(&mut w, &format!("{p}.encoder_attn_layer_norm"), dev)?,
                cross: Attention::load(&mut w, &format!("{p}.encoder_attn"), heads, dev)?,
                mlp_ln: LayerNorm::load(&mut w, &format!("{p}.final_layer_norm"), dev)?,
                mlp: Mlp::load(&mut w, &p, dev)?,
            });
        }
        let dec_ln = LayerNorm::load(&mut w, "decoder.layer_norm", dev)?;

        Ok(Self {
            device: device.clone(),
            conv1_w,
            conv1_b,
            conv2_w,
            conv2_b,
            enc_pos,
            enc_blocks,
            enc_ln,
            tok_emb,
            dec_pos,
            dec_blocks,
            dec_ln,
            n_vocab,
            d_model,
            n_mels,
            n_text_ctx,
        })
    }

    pub fn encode(&self, mel: &[f32]) -> Tensor<2> {
        self.encode_impl(mel, false).0
    }

    pub fn encode_profiled(&self, mel: &[f32]) -> (Tensor<2>, Prof) {
        self.encode_impl(mel, true)
    }

    fn encode_impl(&self, mel: &[f32], on: bool) -> (Tensor<2>, Prof) {
        let mut p = Prof::default();
        let mut lap = Lap {
            on,
            t: std::time::Instant::now(),
        };
        let frames = mel.len() / self.n_mels;
        let x = Tensor::<2>::from_data(
            TensorData::new(mel.to_vec(), [self.n_mels, frames]),
            &self.device,
        )
        .unsqueeze::<3>();
        let x = activation::gelu(conv1d(
            x,
            self.conv1_w.clone(),
            Some(self.conv1_b.clone()),
            ConvOptions::new([1], [1], [1], 1),
        ));
        let x = activation::gelu(conv1d(
            x,
            self.conv2_w.clone(),
            Some(self.conv2_b.clone()),
            ConvOptions::new([2], [1], [1], 1),
        ));
        let [_, d, t] = x.dims();
        let mut x = x.reshape([d, t]).transpose();
        x = x + self.enc_pos.clone().slice([0..t, 0..d]);
        sync2(on, &x);
        lap.tick(&mut p.conv);

        for blk in &self.enc_blocks {
            let h = blk.attn_ln.forward(x.clone());
            sync2(on, &h);
            lap.tick(&mut p.layer_norm);
            let (k, v) = blk.attn.project_kv(&h);
            sync3(on, &k);
            sync3(on, &v);
            lap.tick(&mut p.kv_proj);
            let a = blk.attn.attend(h, k, v, None);
            sync2(on, &a);
            lap.tick(&mut p.attention);
            x = x + a;
            sync2(on, &x);
            lap.tick(&mut p.residual);
            let h = blk.mlp_ln.forward(x.clone());
            sync2(on, &h);
            lap.tick(&mut p.layer_norm);
            let m = blk.mlp.forward(h);
            sync2(on, &m);
            lap.tick(&mut p.mlp);
            x = x.clone() + m;
            sync2(on, &x);
            lap.tick(&mut p.residual);
        }
        let out = self.enc_ln.forward(x);
        sync2(on, &out);
        lap.tick(&mut p.layer_norm);
        (out, p)
    }

    pub fn start_decoding(&self, enc: &Tensor<2>) -> DecoderState {
        DecoderState {
            cross: self
                .dec_blocks
                .iter()
                .map(|b| {
                    let (k, v) = b.cross.project_kv(enc);
                    (contiguous3(k), contiguous3(v))
                })
                .collect(),
            selfkv: self.dec_blocks.iter().map(|_| None).collect(),
            pos: 0,
        }
    }

    pub fn decode_step(&self, st: &mut DecoderState, tokens: &[u32]) -> Result<Vec<f32>> {
        let logits = self.decode_logits(st, tokens)?;
        Ok(logits
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?)
    }

    pub fn decode_step_argmax(
        &self,
        st: &mut DecoderState,
        tokens: &[u32],
        bias: &Tensor<1>,
    ) -> Result<u32> {
        let logits = self.decode_logits(st, tokens)? + bias.clone();
        let idx = logits.argmax(0).into_data().convert::<i64>();
        let v = idx
            .try_to_vec::<i64>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        Ok(v[0] as u32)
    }

    fn decode_logits(&self, st: &mut DecoderState, tokens: &[u32]) -> Result<Tensor<1>> {
        self.decode_logits_impl(st, tokens, None)
    }

    pub fn decode_step_profiled(
        &self,
        st: &mut DecoderState,
        tokens: &[u32],
        prof: &mut DecProf,
    ) -> Result<()> {
        let l = self.decode_logits_impl(st, tokens, Some(prof))?;
        let _ = l.slice([0..1]).into_data();
        Ok(())
    }

    fn decode_logits_impl(
        &self,
        st: &mut DecoderState,
        tokens: &[u32],
        mut prof: Option<&mut DecProf>,
    ) -> Result<Tensor<1>> {
        let on = prof.is_some();
        let mut dp = DecProf::default();
        let mut lap = Lap {
            on,
            t: std::time::Instant::now(),
        };
        let t = tokens.len();
        let past = st.pos;
        if t == 0 || past + t > self.n_text_ctx {
            bail!(
                "decoder context overflow ({} + {} > {})",
                past,
                t,
                self.n_text_ctx
            );
        }
        let d = self.d_model;
        let ids: Vec<i64> = tokens.iter().map(|&v| v as i64).collect();
        let ids = Tensor::<1, Int>::from_data(TensorData::new(ids, [t]), &self.device);
        let mut x = self.tok_emb.clone().select(0, ids)
            + self.dec_pos.clone().slice([past..past + t, 0..d]);
        sync2(on, &x);
        lap.tick(&mut dp.embed);

        let mask = if t > 1 {
            let s = past + t;
            let mut m = vec![0.0f32; t * s];
            for i in 0..t {
                for j in (past + i + 1)..s {
                    m[i * s + j] = 1.0;
                }
            }
            let heads = d / HEAD_DIM;
            let m =
                Tensor::<2>::from_data(TensorData::new(m, [t, s]), &self.device).greater_elem(0.5);
            Some(m.unsqueeze::<4>().expand([1, heads, t, s]))
        } else {
            None
        };

        for (li, blk) in self.dec_blocks.iter().enumerate() {
            let h = blk.attn_ln.forward(x.clone());
            let (k_new, v_new) = blk.attn.project_kv(&h);
            let (k, v) = match st.selfkv[li].take() {
                Some((k_old, v_old)) => (
                    Tensor::cat(vec![k_old, k_new], 1),
                    Tensor::cat(vec![v_old, v_new], 1),
                ),
                None => (k_new, v_new),
            };
            st.selfkv[li] = Some((k.clone(), v.clone()));
            x = x + blk.attn.attend(h, k, v, mask.clone());
            sync2(on, &x);
            lap.tick(&mut dp.self_attn);

            let h = blk.cross_ln.forward(x.clone());
            let (ck, cv) = st.cross[li].clone();
            x = x + blk.cross.attend(h, ck, cv, None);
            sync2(on, &x);
            lap.tick(&mut dp.cross_attn);

            x = x.clone() + blk.mlp.forward(blk.mlp_ln.forward(x));
            sync2(on, &x);
            lap.tick(&mut dp.mlp);
        }
        st.pos += t;

        let x = self.dec_ln.forward(x);
        let last = x.slice([t - 1..t, 0..d]);
        let logits = self.tok_emb.clone().matmul(last.transpose());
        let logits = logits.reshape([self.n_vocab]);
        if on {
            let _ = logits.clone().slice([0..1]).into_data();
        }
        lap.tick(&mut dp.logits);
        if let Some(p) = prof.as_deref_mut() {
            p.embed += dp.embed;
            p.self_attn += dp.self_attn;
            p.cross_attn += dp.cross_attn;
            p.mlp += dp.mlp;
            p.logits += dp.logits;
        }
        Ok(logits)
    }
}
