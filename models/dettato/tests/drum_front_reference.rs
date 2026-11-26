use dettato_rs::drums::features;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    sample_rate: usize,
    samples: usize,
    frequencies: [f64;3],
    amplitudes: [f64;3],
    frames: usize,
    points: Vec<[f64;3]>,
}

#[test]
fn drum_feature_subset_matches_upstream_python_fixture() {
    let fixture:Fixture=serde_json::from_str(include_str!("fixtures/drum_front_reference.json")).unwrap();
    let audio=(0..fixture.samples).map(|i| {
        let t=i as f64/fixture.sample_rate as f64;
        fixture.frequencies.iter().zip(fixture.amplitudes).map(|(f,a)|
            a*(2.0*std::f64::consts::PI*(*f)*t).sin()).sum::<f64>() as f32
    }).collect::<Vec<_>>();
    let result=features::extract(&audio).unwrap();
    assert_eq!(result.frames,fixture.frames);
    assert_eq!(result.data.len(),fixture.frames*280);
    for [time,band,expected] in fixture.points {
        let index=time as usize*280+band as usize;
        let got=result.data[index];
        let tol=if band>=128.0&&band<160.0 {0.04}else{0.35};
        assert!((f64::from(got)-expected).abs()<tol,
            "drum frontend t={time} band={band}: got {got}, expected {expected}");
    }
}
