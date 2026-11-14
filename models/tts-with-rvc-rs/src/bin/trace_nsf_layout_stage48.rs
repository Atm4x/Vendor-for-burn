#[cfg(feature = "cpu")]
use burn::tensor::{Device, Tensor, TensorData};
#[cfg(feature = "cpu")]
use tts_with_rvc_rs::{
    checkpoint::RvcCheckpoint,
    decoder::{NsfDecoderFixture, ResidualExecution},
};

#[cfg(feature = "cpu")]
fn main() {
    let fixture: NsfDecoderFixture =
        serde_json::from_str(include_str!("../../tests/fixtures/nsf_decoder.json")).unwrap();
    let checkpoint = RvcCheckpoint::open(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/synthetic_nsf_decoder.pth"
    ))
    .unwrap();
    let device = Device::cpu();
    let decoder = fixture.load(&checkpoint, &device).unwrap();
    let requested: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("frames must be positive integers"))
        .collect();
    let frames = if requested.is_empty() {
        vec![4, 64, 256]
    } else {
        requested
    };
    let [batch, channels, original] = fixture.input_shape;
    let upsample = fixture.source_shape[2] / original;
    assert_eq!(batch, 1);
    for count in frames {
        assert!(count > 0);
        let original_input = &fixture.input;
        let values: Vec<f32> = (0..channels)
            .flat_map(|c| (0..count).map(move |t| original_input[c * original + t % original]))
            .collect();
        let harmonic: Vec<f32> = (0..count * upsample)
            .map(|t| fixture.harmonic_source[t % fixture.source_shape[2]])
            .collect();
        let input = Tensor::from_data(TensorData::new(values, [1, channels, count]), &device);
        let source =
            Tensor::from_data(TensorData::new(harmonic, [1, 1, count * upsample]), &device);
        let condition = fixture.condition.as_ref().map(|v| {
            Tensor::from_data(
                TensorData::new(v.clone(), fixture.condition_shape.unwrap()),
                &device,
            )
        });
        eprintln!("CUBECL_NSF_BEGIN frames={count}");
        let output = decoder
            .forward_with_execution(input, source, condition, ResidualExecution::Sequential)
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();
        eprintln!("CUBECL_NSF_END frames={count}");
        let sum: f64 = output.iter().map(|v| *v as f64).sum();
        println!(
            "frames={count} output_elements={} checksum_sum={sum:.12}",
            output.len()
        );
    }
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("Use --no-default-features --features cpu");
    std::process::exit(2);
}
