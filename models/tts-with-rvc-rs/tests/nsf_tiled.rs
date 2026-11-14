#![cfg(feature = "cpu")]

use burn::tensor::{Device, Tensor, TensorData};
use tts_with_rvc_rs::{checkpoint::RvcCheckpoint, decoder::NsfDecoderFixture};

#[test]
fn full_nsf_matches_output_tiled_at_multiple_core_lengths() {
    let fixture: NsfDecoderFixture =
        serde_json::from_str(include_str!("fixtures/nsf_decoder.json")).unwrap();
    let checkpoint = RvcCheckpoint::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/synthetic_nsf_decoder.pth"
    ))
    .unwrap();
    let device = Device::cpu();
    let model = fixture.load(&checkpoint, &device).unwrap();
    let [batch, channels, initial_frames] = fixture.input_shape;
    let total_frames = 48;
    let rate: usize = fixture.upsample_rates.iter().product();
    let mut latent = Vec::new();
    let mut harmonic = Vec::new();
    for channel in 0..channels {
        for frame in 0..total_frames {
            let source = fixture.input[channel * initial_frames + frame % initial_frames];
            latent.push(source * (1.0 + ((frame * 3 + channel * 7) % 13) as f32 / 91.0));
        }
    }
    for sample in 0..total_frames * rate {
        harmonic.push((sample as f32 * 0.019).sin() * 0.3);
    }
    let input = Tensor::from_data(
        TensorData::new(latent, [batch, channels, total_frames]),
        &device,
    );
    let source = Tensor::from_data(
        TensorData::new(harmonic, [batch, 1, total_frames * rate]),
        &device,
    );
    let condition = fixture.condition.as_ref().map(|values| {
        Tensor::from_data(
            TensorData::new(values.clone(), fixture.condition_shape.unwrap()),
            &device,
        )
    });
    let baseline = model
        .forward(input.clone(), source.clone(), condition.clone())
        .into_data()
        .try_into_vec::<f32>()
        .unwrap();
    for tile_frames in [3, 7, 13, 24, 31] {
        let parts = model
            .tiling_geometry()
            .plan(total_frames, tile_frames)
            .unwrap();
        assert_eq!(parts.len(), total_frames.div_ceil(tile_frames));
        let candidate = model
            .forward_tiled(
                input.clone(),
                source.clone(),
                condition.clone(),
                tile_frames,
            )
            .unwrap()
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();
        assert_eq!(baseline.len(), candidate.len());
        let max_abs = baseline
            .iter()
            .zip(candidate.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-5, "tile={tile_frames} max_abs={max_abs}");
    }
}
