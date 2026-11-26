use crate::{Result, checkpoint::Checkpoint};

#[derive(Debug, Clone)]
pub(crate) enum ResidualWeights {
    Grn { gamma: Vec<f32>, beta: Vec<f32> },
    LayerScale(Vec<f32>),
}

impl ResidualWeights {
    pub(crate) fn load(ckpt: &Checkpoint, prefix: &str, dim: usize, hidden: usize) -> Result<Self> {
        let gamma = format!("{prefix}.grn.gamma");
        let beta = format!("{prefix}.grn.beta");
        if ckpt.tensors().contains_key(&gamma) || ckpt.tensors().contains_key(&beta) {
            return Ok(Self::Grn {
                gamma: ckpt.read_f32(&gamma, &[1, 1, hidden])?,
                beta: ckpt.read_f32(&beta, &[1, 1, hidden])?,
            });
        }
        Ok(Self::LayerScale(
            ckpt.read_f32(&format!("{prefix}.gamma"), &[dim])?,
        ))
    }
}
