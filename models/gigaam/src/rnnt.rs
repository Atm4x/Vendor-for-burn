use crate::audio::Features;
use crate::model::{Config, EncoderModel};
use crate::weights::Weights;
use anyhow::{Context, Result, ensure};
use burn::tensor::{Device, Tensor, activation};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RnntConfig {
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    pub joint_hidden: usize,
    #[serde(default = "default_max_symbols")]
    pub max_symbols_per_step: usize,
}
fn default_max_symbols() -> usize {
    10
}
impl RnntConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.pred_hidden > 0 && self.pred_rnn_layers > 0 && self.joint_hidden > 0,
            "invalid RNN-T hidden dimensions"
        );
        ensure!(
            (1..=100).contains(&self.max_symbols_per_step),
            "max_symbols_per_step must be between 1 and 100"
        );
        Ok(())
    }
}

struct Projection {
    weight: Tensor<2>,
    bias: Tensor<1>,
}
impl Projection {
    fn load(w: &mut Weights, prefix: &str, device: &Device) -> Result<Self> {
        Ok(Self {
            weight: w.tensor(&format!("{prefix}.weight"), device)?,
            bias: w.tensor(&format!("{prefix}.bias"), device)?,
        })
    }
    fn apply(&self, input: Tensor<2>) -> Tensor<2> {
        input.matmul(self.weight.clone().transpose())
            + self.bias.clone().reshape([1, self.bias.dims()[0]])
    }
    fn input_dim(&self) -> usize {
        self.weight.dims()[1]
    }
    fn output_dim(&self) -> usize {
        self.weight.dims()[0]
    }
}

struct LstmLayer {
    weight_ih: Tensor<2>,
    weight_hh: Tensor<2>,
    bias: Tensor<1>,
}
impl LstmLayer {
    fn load(w: &mut Weights, base: &str, index: usize, device: &Device, hidden: usize) -> Result<Self> {
        let weight_ih = w.tensor(&format!("{base}.weight_ih_l{index}"), device)?;
        let weight_hh = w.tensor(&format!("{base}.weight_hh_l{index}"), device)?;
        let b_ih: Tensor<1> = w.tensor(&format!("{base}.bias_ih_l{index}"), device)?;
        let b_hh: Tensor<1> = w.tensor(&format!("{base}.bias_hh_l{index}"), device)?;
        ensure!(
            weight_ih.dims() == [hidden * 4, hidden] && weight_hh.dims() == [hidden * 4, hidden],
            "invalid LSTM layer {index} weights"
        );
        Ok(Self {
            weight_ih,
            weight_hh,
            bias: b_ih + b_hh,
        })
    }
    fn forward(&self, x: Tensor<2>, h: Tensor<2>, c: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let hidden = h.dims()[1];
        let gates = x.matmul(self.weight_ih.clone().transpose())
            + h.matmul(self.weight_hh.clone().transpose())
            + self.bias.clone().reshape([1, hidden * 4]);
        let chunk = |start| gates.clone().slice([0..1, start..start + hidden]);
        let i = activation::sigmoid(chunk(0));
        let f = activation::sigmoid(chunk(hidden));
        let g = chunk(hidden * 2).tanh();
        let o = activation::sigmoid(chunk(hidden * 3));
        let c = f * c + i * g;
        let h = o * c.clone().tanh();
        (h, c)
    }
}

struct Predictor {
    embeddings: Tensor<2>,
    layers: Vec<LstmLayer>,
    hidden: usize,
}

struct PredictorState {
    h: Vec<Tensor<2>>,
    c: Vec<Tensor<2>>,
}
impl Predictor {
    fn load(w: &mut Weights, prefix: &str, config: &RnntConfig, classes: usize, device: &Device) -> Result<Self> {
        let embeddings = w.checked_tensor(&format!("{prefix}.embed.weight"), [classes,config.pred_hidden], device)?;
        ensure!(
            embeddings.dims() == [classes, config.pred_hidden],
            "RNN-T embedding tensor mismatch"
        );
        let layers = (0..config.pred_rnn_layers)
            .map(|idx| LstmLayer::load(w, &format!("{prefix}.lstm"), idx, device, config.pred_hidden))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embeddings,
            layers,
            hidden: config.pred_hidden,
        })
    }

    fn predict(
        &self,
        label: Option<usize>,
        state: Option<&PredictorState>,
        device: &Device,
    ) -> Result<(Tensor<2>, PredictorState)> {
        let mut x = match label {
            Some(id) => {
                ensure!(id < self.embeddings.dims()[0], "RNN-T label out of range");
                self.embeddings.clone().slice([id..id + 1, 0..self.hidden])
            }
            None => Tensor::<2>::zeros([1, self.hidden], device),
        };
        let mut hs = Vec::with_capacity(self.layers.len());
        let mut cs = Vec::with_capacity(self.layers.len());
        for (idx, layer) in self.layers.iter().enumerate() {
            let h = state.map_or_else(
                || Tensor::<2>::zeros([1, self.hidden], device),
                |s| s.h[idx].clone(),
            );
            let c = state.map_or_else(
                || Tensor::<2>::zeros([1, self.hidden], device),
                |s| s.c[idx].clone(),
            );
            let (next_h, next_c) = layer.forward(x, h, c);
            x = next_h.clone();
            hs.push(next_h);
            cs.push(next_c);
        }
        Ok((x, PredictorState { h: hs, c: cs }))
    }
}

struct Joint {
    enc: Projection,
    pred: Projection,
    output: Projection,
}
impl Joint {
    fn load(w: &mut Weights, prefix: &str, width: usize, classes: usize, rnnt: &RnntConfig, device: &Device) -> Result<Self> {
        let enc = Projection::load(w, &format!("{prefix}.enc"), device)?;
        let pred = Projection::load(w, &format!("{prefix}.pred"), device)?;
        let output = Projection::load(w, &format!("{prefix}.joint_net.1"), device)?;
        ensure!(
            enc.input_dim() == width && enc.output_dim() == rnnt.joint_hidden,
            "RNN-T encoder projection mismatch"
        );
        ensure!(
            pred.input_dim() == rnnt.pred_hidden && pred.output_dim() == rnnt.joint_hidden,
            "RNN-T predictor projection mismatch"
        );
        ensure!(
            output.input_dim() == rnnt.joint_hidden && output.output_dim() == classes,
            "RNN-T joint classifier mismatch"
        );
        Ok(Self { enc, pred, output })
    }
    fn scores(&self, enc: Tensor<2>, pred: Tensor<2>) -> Tensor<2> {
        self.output
            .apply(activation::relu(enc + self.pred.apply(pred)))
    }
}

/// Encoder-independent RNN-T head; every decode owns its predictor state.
pub struct RnntHead {
    predictor: Predictor,
    joint: Joint,
    classes: usize,
    max_symbols_per_step: usize,
}
#[derive(Default)]
pub(crate) struct RnntDecodeState {
    last_label: Option<usize>,
    hidden: Option<PredictorState>,
    prediction: Option<(Tensor<2>, PredictorState)>,
    frame_offset: usize,
}

impl RnntHead {
    pub fn load(w: &mut Weights, prefix: &str, width: usize, classes: usize, rnnt: &RnntConfig, device: &Device) -> Result<Self> {
        rnnt.validate()?;
        ensure!(width > 0 && classes > 1, "invalid RNN-T head geometry");
        let predictor = Predictor::load(w, &format!("{prefix}.decoder"), rnnt, classes, device)?;
        let joint = Joint::load(w, &format!("{prefix}.joint"), width, classes, rnnt, device)?;
        Ok(Self { predictor, joint, classes, max_symbols_per_step: rnnt.max_symbols_per_step })
    }

    pub fn decode_encoded(
        &self,
        encoded: Tensor<3>,
        device: &Device,
    ) -> Result<(Vec<usize>, Vec<usize>, usize)> {
        self.decode_chunk(encoded, device, &mut RnntDecodeState::default())
    }

    pub(crate) fn decode_chunk(
        &self, encoded: Tensor<3>, device: &Device, state: &mut RnntDecodeState,
    ) -> Result<(Vec<usize>, Vec<usize>, usize)> {
        let [batch, frames, dim] = encoded.dims();
        ensure!(batch == 1, "RNN-T decoding expects one encoder sequence");
        ensure!(dim == self.joint.enc.input_dim(), "RNN-T encoder width mismatch");
        if frames == 0 { return Ok((Vec::new(), Vec::new(), 0)); }
        let enc_proj = self.joint.enc.apply(encoded.reshape([frames, dim]));
        let mut ids = Vec::new();
        let mut token_frames = Vec::new();
        let blank = self.classes - 1;
        let end = state.frame_offset.checked_add(frames).context("RNN-T frame offset overflow")?;
        for frame in 0..frames {
            let enc_frame = enc_proj
                .clone()
                .slice([frame..frame + 1, 0..self.joint.enc.output_dim()]);
            for _ in 0..self.max_symbols_per_step {
                if state.prediction.is_none() {
                    state.prediction = Some(self.predictor.predict(state.last_label, state.hidden.as_ref(), device)?);
                }
                let (pred, _) = state.prediction.as_ref().unwrap();
                let logits = self.joint.scores(enc_frame.clone(), pred.clone());
                let values = logits
                    .into_data()
                    .try_to_vec::<f32>()
                    .context("reading RNN-T probabilities")?;
                let best = values
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                    .context("empty RNN-T joint output")?
                    .0;
                if best == blank {
                    break;
                }
                ids.push(best);
                token_frames.push(state.frame_offset + frame);
                state.last_label = Some(best);
                state.hidden = state.prediction.take().map(|(_, next)| next);
            }
        }
        state.frame_offset = end;
        Ok((ids, token_frames, frames))
    }
}

pub struct RnntModel {
    encoder: EncoderModel,
    head: RnntHead,
}
impl RnntModel {
    pub fn load(w: &mut Weights, config: Config, rnnt: RnntConfig, device: &Device) -> Result<Self> {
        config.validate()?;
        let head = RnntHead::load(w, "head", config.d_model, config.num_classes, &rnnt, device)?;
        let encoder = EncoderModel::load(w, config, device)?;
        Ok(Self { encoder, head })
    }

    pub fn n_mels(&self) -> usize {
        self.encoder.n_mels()
    }

    pub fn encode(&self, features: &Features, device: &Device) -> Result<Tensor<3>> {
        self.encoder.encode(features, device)
    }
    pub fn encode_padded(
        &self,
        features: &[Features],
        device: &Device,
    ) -> Result<(Tensor<3>, Vec<usize>)> {
        self.encoder.encode_padded(features, device)
    }

    pub fn infer_ids(
        &self,
        features: &Features,
        device: &Device,
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        let (ids, token_frames, _) = self.infer_ids_with_length(features, device)?;
        Ok((ids, token_frames))
    }

    pub fn infer_ids_with_length(
        &self,
        features: &Features,
        device: &Device,
    ) -> Result<(Vec<usize>, Vec<usize>, usize)> {
        let encoded = self.encoder.encode(features, device)?;
        self.head.decode_encoded(encoded, device)
    }

    pub fn infer_ids_padded(
        &self,
        features: &[Features],
        device: &Device,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>, usize)>> {
        let (encoded, lengths) = self.encoder.encode_padded(features, device)?;
        let [batch, _, dim] = encoded.dims();
        let mut results = Vec::with_capacity(batch);
        for (index, &frames) in lengths.iter().enumerate() {
            let sample = encoded.clone().slice([index..index + 1, 0..frames, 0..dim]);
            results.push(self.head.decode_encoded(sample, device)?);
        }
        Ok(results)
    }


}

#[cfg(all(test, any(feature = "cpu", feature = "flex")))]
mod tests {
    use super::*;
    use burn::tensor::TensorData;

    fn tensor<const D: usize>(values: Vec<f32>, shape: [usize; D], device: &Device) -> Tensor<D> {
        Tensor::from_data(TensorData::new(values, shape), device)
    }

    fn linspace(start: f32, end: f32, count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| start + (end - start) * (i as f32) / (count - 1) as f32)
            .collect()
    }

    #[cfg(feature = "cpu")]
    fn test_device() -> Device {
        Device::cpu()
    }

    #[cfg(all(not(feature = "cpu"), feature = "flex"))]
    fn test_device() -> Device {
        Device::flex()
    }

    #[test]
    fn lstm_matches_pytorch_two_recurrent_steps() {
        let d = test_device();
        let hidden = 3;
        let layer = LstmLayer {
            weight_ih: tensor(linspace(-0.12, 0.15, 36), [12, 3], &d),
            weight_hh: tensor(linspace(0.07, -0.09, 36), [12, 3], &d),
            bias: tensor(
                linspace(-0.02, 0.03, hidden * 4)
                    .iter()
                    .zip(linspace(0.03, -0.01, hidden * 4))
                    .map(|(a, b)| a + b)
                    .collect(),
                [hidden * 4],
                &d,
            ),
        };
        let mut h = tensor(vec![0.03, 0.0, -0.02], [1, 3], &d);
        let mut c = tensor(vec![0.01, -0.06, 0.02], [1, 3], &d);
        let steps = [
            (
                vec![0.15, -0.12, 0.04],
                [0.00672567263, -0.01048186049, 0.01052685454],
                [0.0132945925, -0.0206967015, 0.0207610559],
            ),
            (
                vec![-0.03, 0.08, 0.11],
                [0.00862589106, 0.00110267068, 0.01293370686],
                [0.01696329377, 0.00216345396, 0.02532536909],
            ),
        ];
        for (x, expected_h, expected_c) in steps {
            (h, c) = layer.forward(tensor(x, [1, 3], &d), h, c);
            for (actual, expected) in h
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .zip(expected_h)
            {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "LSTM h mismatch: {actual} != {expected}"
                );
            }
            for (actual, expected) in c
                .clone()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .zip(expected_c)
            {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "LSTM c mismatch: {actual} != {expected}"
                );
            }
        }
    }
}
