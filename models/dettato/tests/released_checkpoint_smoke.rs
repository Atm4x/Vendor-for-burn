use dettato_rs::{checkpoint::{Checkpoint,ModelPaths}, pipeline::Dettato};
use burn::tensor::Device;

#[test]
#[ignore = "set DETTATO_WEIGHTS to the unpacked archive and run --ignored"]
fn four_released_checkpoints_can_initialize_pipeline() {
    let root=std::env::var("DETTATO_WEIGHTS").expect("DETTATO_WEIGHTS path required");
    for stage in ["transcriber","evidence","grouper","drums"] {
        let path=ModelPaths::in_directory(&root,stage).unwrap();
        let config=path.load_config().unwrap();
        let checkpoint=Checkpoint::open(&path.weights).unwrap();
        assert!(!checkpoint.names().unwrap().is_empty());
        match stage {
            "drums"=>assert_eq!(config["model_type"],"dettato-drums"),
            _=>assert_eq!(config["stage"],stage),
        }
    }
    let device=Device::default();
    let _=Dettato::from_directory(&root,&device).unwrap();
}

#[test]
#[ignore = "set DETTATO_WEIGHTS to the unpacked archive; slow CPU/GPU Burn test"]
fn drum_ensemble_matches_upstream_logit_fixture() {
    use dettato_rs::drums::{DrumTranscriber,features};
    use serde_json::Value;

    let root=std::env::var("DETTATO_WEIGHTS").expect("DETTATO_WEIGHTS path required");
    let d=DrumTranscriber::from_directory(&root,&Device::default()).unwrap();
    let rate=44100.0;
    let audio=(0..(rate*0.22) as usize).map(|i| {
        let t=i as f64/rate;
        (0.19*(2.0*std::f64::consts::PI*440.0*t).sin()
          +0.12*(2.0*std::f64::consts::PI*880.0*t).sin()
          +0.02*(2.0*std::f64::consts::PI*4310.0*t).sin()) as f32
    }).collect::<Vec<_>>();
    let p=d.probabilities(&features::extract(&audio).unwrap()).unwrap();
    let fixture:Value=serde_json::from_str(include_str!("fixtures/drum_logits_reference.json")).unwrap();
    assert_eq!(p.frames,fixture["frames"].as_u64().unwrap() as usize);
    let router=fixture["router"].as_f64().unwrap();
    assert!((p.router as f64-router).abs()<0.1,
        "router differed: got {} expected {router}",p.router);
    let selected=&fixture["selected"];
    for (name,channels,data) in [
        ("kit",61,&p.kit), ("kit_vel",61,&p.kit_velocity),
        ("drum",61,&p.orchestral), ("timpani",13,&p.timpani),
        ("vel",61,&p.orchestral_velocity),
    ] {
        for sample in selected[name].as_array().unwrap() {
            let t=sample[0].as_u64().unwrap() as usize;
            let c=sample[1].as_u64().unwrap() as usize;
            let expected=sample[2].as_f64().unwrap();
            let got=f64::from(data[t*channels+c]);
            assert!((got-expected).abs()<0.1,
                "{name} frame={t} channel={c}: got {got} expected {expected}");
        }
    }
}
