use burn::tensor::{Device, Tensor, activation};

use crate::{checkpoint::RvcCheckpoint, rmvpe::RmvpeHead, rmvpe_unet::RmvpeUnet};

struct GruDirection {
    input_weight: Tensor<2>,
    hidden_weight: Tensor<2>,
    input_bias: Tensor<1>,
    hidden_bias: Tensor<1>,
}

impl GruDirection {
    fn load(checkpoint: &RvcCheckpoint, suffix: &str, device: &Device) -> Result<Self, String> {
        let prefix = "fc.0.gru";
        let input_weight =
            checkpoint.tensor::<2>(&format!("{prefix}.weight_ih_l0{suffix}"), device)?;
        let hidden_weight =
            checkpoint.tensor::<2>(&format!("{prefix}.weight_hh_l0{suffix}"), device)?;
        let input_bias = checkpoint.tensor::<1>(&format!("{prefix}.bias_ih_l0{suffix}"), device)?;
        let hidden_bias =
            checkpoint.tensor::<1>(&format!("{prefix}.bias_hh_l0{suffix}"), device)?;
        let [gates, inputs] = input_weight.dims();
        let [hidden_gates, hidden] = hidden_weight.dims();
        if hidden == 0
            || inputs == 0
            || gates != 3 * hidden
            || hidden_gates != gates
            || input_bias.dims() != [gates]
            || hidden_bias.dims() != [gates]
        {
            return Err(format!(
                "{prefix}{suffix}: incompatible GRU gate tensor dimensions"
            ));
        }
        Ok(Self {
            input_weight,
            hidden_weight,
            input_bias,
            hidden_bias,
        })
    }

    fn input_size(&self) -> usize {
        self.input_weight.dims()[1]
    }
    fn hidden_size(&self) -> usize {
        self.hidden_weight.dims()[1]
    }

    fn forward(&self, input: Tensor<3>, reverse: bool) -> Tensor<3> {
        let [batch, time, features] = input.dims();
        let hidden_size = self.hidden_size();
        assert_eq!(
            features,
            self.input_size(),
            "RMVPE GRU input features mismatch"
        );
        assert!(time > 0, "RMVPE GRU requires a nonempty time dimension");
        let mut hidden = Tensor::zeros([batch, hidden_size], &input.device());
        let hidden_weight = self.hidden_weight.clone().transpose();
        let hidden_bias = self.hidden_bias.clone().reshape([1, 3 * hidden_size]);
        let input_gates = (input
            .reshape([batch * time, features])
            .matmul(self.input_weight.clone().transpose())
            + self.input_bias.clone().reshape([1, 3 * hidden_size]))
        .reshape([batch, time, 3 * hidden_size]);
        let mut outputs = Vec::with_capacity(time);

        for step in 0..time {
            let frame = if reverse { time - 1 - step } else { step };
            let input_gates = input_gates
                .clone()
                .slice([0..batch, frame..frame + 1, 0..3 * hidden_size])
                .reshape([batch, 3 * hidden_size]);
            let hidden_gates = hidden.clone().matmul(hidden_weight.clone()) + hidden_bias.clone();
            let ir = input_gates.clone().slice([0..batch, 0..hidden_size]);
            let iz = input_gates
                .clone()
                .slice([0..batch, hidden_size..2 * hidden_size]);
            let inn = input_gates.slice([0..batch, 2 * hidden_size..3 * hidden_size]);
            let hr = hidden_gates.clone().slice([0..batch, 0..hidden_size]);
            let hz = hidden_gates
                .clone()
                .slice([0..batch, hidden_size..2 * hidden_size]);
            let hn = hidden_gates.slice([0..batch, 2 * hidden_size..3 * hidden_size]);
            let reset = activation::sigmoid(ir + hr);
            let update = activation::sigmoid(iz + hz);
            let candidate = activation::tanh(inn + reset * hn);
            hidden = (update.clone() * -1.0 + 1.0) * candidate + update * hidden;
            outputs.push(hidden.clone().reshape([batch, 1, hidden_size]));
        }
        if reverse {
            outputs.reverse();
        }
        Tensor::cat(outputs, 1)
    }
}

pub struct RmvpeBiGru {
    forward_direction: GruDirection,
    reverse_direction: GruDirection,
}

impl RmvpeBiGru {
    pub fn load(checkpoint: &RvcCheckpoint, device: &Device) -> Result<Self, String> {
        let forward_direction = GruDirection::load(checkpoint, "", device)?;
        let reverse_direction = GruDirection::load(checkpoint, "_reverse", device)?;
        if forward_direction.input_size() != reverse_direction.input_size()
            || forward_direction.hidden_size() != reverse_direction.hidden_size()
        {
            return Err("RMVPE bidirectional GRU dimensions differ".into());
        }
        Ok(Self {
            forward_direction,
            reverse_direction,
        })
    }

    pub fn input_size(&self) -> usize {
        self.forward_direction.input_size()
    }
    pub fn hidden_size(&self) -> usize {
        self.forward_direction.hidden_size()
    }

    pub fn forward(&self, input: Tensor<3>) -> Tensor<3> {
        Tensor::cat(
            vec![
                self.forward_direction.forward(input.clone(), false),
                self.reverse_direction.forward(input, true),
            ],
            2,
        )
    }
}

pub struct RmvpeNetwork {
    unet: RmvpeUnet,
    gru: RmvpeBiGru,
    head: RmvpeHead,
}

impl RmvpeNetwork {
    pub fn load(
        checkpoint: &RvcCheckpoint,
        encoder_levels: usize,
        intermediate_levels: usize,
        mel_bins: usize,
        device: &Device,
    ) -> Result<Self, String> {
        if mel_bins == 0 || mel_bins % (1 << encoder_levels) != 0 {
            return Err("RMVPE mel bins must be divisible by the U-Net downsampling factor".into());
        }
        let unet = RmvpeUnet::load(checkpoint, encoder_levels, intermediate_levels, device)?;
        let gru = RmvpeBiGru::load(checkpoint, device)?;
        if gru.input_size() != 3 * mel_bins || gru.hidden_size() != 256 {
            return Err("RMVPE GRU/CNN/F0 projection channel mismatch".into());
        }
        let head = RmvpeHead::load(checkpoint, device)?;
        Ok(Self { unet, gru, head })
    }

    pub fn forward(&self, mel: Tensor<3>) -> Tensor<3> {
        let [batch, bins, frames] = mel.dims();
        let factor = 1 << self.unet.encoder_levels();
        assert_eq!(self.gru.input_size(), 3 * bins, "RMVPE mel bins mismatch");
        assert!(
            frames > 0 && frames % factor == 0,
            "RMVPE mel frames must be padded to a multiple of the downsampling factor"
        );
        let device = mel.device();
        let input = mel.swap_dims(1, 2).reshape([batch, 1, frames, bins]);
        let started = std::time::Instant::now();
        let unet = self.unet.forward_gru_features(input);
        crate::diagnostic::measure(&device, "rmvpe.unet", started, serde_json::json!({"output_shape":unet.dims()}));
        let started = std::time::Instant::now();
        let gru = self.gru.forward(unet);
        crate::diagnostic::measure(&device, "rmvpe.bidirectional_gru", started, serde_json::json!({"output_shape":gru.dims()}));
        let started = std::time::Instant::now();
        let out = self.head.forward(gru);
        crate::diagnostic::measure(&device, "rmvpe.f0_head", started, serde_json::json!({"output_shape":out.dims()}));
        out
    }
}
