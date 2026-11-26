use burn::tensor::Device;
use dettato_rs::{
    evidence::{features::EVIDENCE_FEATURES, model::EvidenceNet, repair::RepairInput},
    grouper::net::{GROUPER_INSTRUMENT_COUNT, NoteNet, TRANSCRIBER_TAP_DIM, WindowInput},
};

fn weights() -> String {
    std::env::var("DETTATO_WEIGHTS").expect("DETTATO_WEIGHTS must point to released weights")
}

#[test]
#[ignore = "real released checkpoint inference; set DETTATO_WEIGHTS"]
fn evidence_repairer_accepts_released_checkpoint() {
    let net = EvidenceNet::from_directory(weights(), &Device::default()).unwrap();
    let notes = [RepairInput {
        onset: 0.25,
        offset: 0.65,
        pitch: 69,
        velocity: 96,
        original_index: 0,
    }];
    let evidence = [[0.0f32; EVIDENCE_FEATURES]];
    let result = net.infer(&notes, &evidence, 0.0).unwrap();
    assert_eq!(result.notes.len(), 1);
    assert!(result.notes[0].keep.is_finite());
    assert_eq!(result.notes[0].onset.len(), 201);
    assert_eq!(result.notes[0].duration_ratio.len(), 161);
    for insertion in &result.notes[0].insertion {
        assert_eq!(insertion.pitch.len(), 49);
        assert_eq!(insertion.onset.len(), 401);
        assert_eq!(insertion.duration.len(), 128);
        assert!(insertion.exists.is_finite());
    }
}

#[test]
#[ignore = "real released checkpoint inference; set DETTATO_WEIGHTS"]
fn grouper_notenet_accepts_released_checkpoint() {
    let net = NoteNet::from_directory(weights(), &Device::default()).unwrap();
    let sample_rate = 16_000;
    let audio16k = (0..sample_rate * 2)
        .map(|i| {
            0.08 * (2.0f32 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
        })
        .collect();
    let mut note = [0.0f32; 12];
    note[0] = 0.25;
    note[1] = 0.75;
    note[2] = 69.0;
    note[3] = 1.0;
    note[4] = 0.95;
    note[5] = 0.9;
    let window = WindowInput {
        audio16k,
        notes: vec![note; 4],
        taps: vec![[[0.0f32; TRANSCRIBER_TAP_DIM]; 3]; 4],
    };
    let result = net.infer(&window).unwrap();
    assert_eq!(result.probabilities.len(), 4);
    assert_eq!(result.embeddings.len(), 4);
    assert_eq!(result.hidden.len(), 4);
    assert_eq!(result.keep_logits.len(), 4);
    for row in &result.probabilities {
        assert_eq!(row.len(), GROUPER_INSTRUMENT_COUNT);
        assert!(row.iter().all(|x| x.is_finite()));
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-3);
    }
    for embedding in &result.embeddings {
        assert_eq!(embedding.len(), net.config.embedding);
        assert!(embedding.iter().all(|x| x.is_finite()));
    }
}

#[test]
#[ignore = "real released checkpoint inference; set DETTATO_WEIGHTS"]
fn transcriber_lanenet_accepts_released_checkpoint() {
    use dettato_rs::{
        checkpoint::{Checkpoint, ModelPaths},
        hcqt::Hcqt,
        transcriber::net::{LaneNet, Settings},
    };
    let root = weights();
    let paths = ModelPaths::in_directory(root, "transcriber").unwrap();
    let config = paths.load_config().unwrap();
    let settings = Settings::from_config(&config).unwrap();
    let checkpoint = Checkpoint::open(&paths.weights).unwrap();
    let model = LaneNet::load(&checkpoint, &Device::default(), settings).unwrap();
    let front = Hcqt::default_16k();
    let audio = vec![0.0f32; 8 * 16_000];
    let features = front.extract(&audio);
    let result = model.infer_features(&features).unwrap();
    assert_eq!(result.maps.frames(), 400);
    for head in &result.maps.heads {
        assert_eq!(head.len(), 400 * 88);
        assert!(head.iter().all(|v| v.is_finite()));
    }
}
