use dettato_rs::{
    grouper::front::{AudioFeatures, harmonic_features, pcen, standardise},
    hcqt::Features,
};
use serde_json::Value;

fn floats(row: &Value) -> Vec<f32> {
    row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

#[test]
fn pcen_and_standardise_match_original_pytorch() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/grouper_front_reference.json")).unwrap();
    let p = &fixture["pcen"];
    let mut features = Features {
        channels: p["channels"].as_u64().unwrap() as usize,
        frames: p["frames"].as_u64().unwrap() as usize,
        bins: p["bins"].as_u64().unwrap() as usize,
        data: floats(&p["input"]),
    };
    pcen(&mut features, 0.025, 0.98, 2.0, 0.5, 1e-6);
    standardise(&mut features);
    let expected = floats(&p["expected"]);
    let max_error = features.data.iter().zip(expected.iter())
        .map(|(a, b)| (a-b).abs()).fold(0.0f32, f32::max);
    assert!(max_error < 3e-4, "Grouper PCEN deviation: {max_error}");
}

#[test]
fn harmonic_demixing_matches_original_pytorch() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/grouper_front_reference.json")).unwrap();
    let p = &fixture["hdemix"];
    let t = p["frames"].as_u64().unwrap() as usize;
    let ne = p["extended_bins"].as_u64().unwrap() as usize;
    let power = (0..ne).flat_map(|bin| (0..t).map(move |frame| {
        (1 + (bin * 17 + frame * 13) % 71) as f32 / 100.0
    })).collect();
    let front = AudioFeatures {
        spectrogram: Features { channels: 9, frames: t, bins: 288, data: vec![0.0;9*t*288] },
        extended_bins: ne,
        low_shift: p["low_shift"].as_i64().unwrap() as isize,
        power,
    };
    let notes = p["notes"].as_array().unwrap().iter().map(|row| {
        let vec = floats(row);
        <[f32;12]>::try_from(vec).unwrap()
    }).collect::<Vec<_>>();
    let prior = floats(&p["prior"]);
    let result = harmonic_features(&front, &notes, &prior, 4).unwrap();
    let expected = p["expected"].as_array().unwrap();
    for (i, (got, reference)) in result.iter().zip(expected).enumerate() {
        let diff = got.iter().zip(floats(reference)).map(|(a, b)| (a-b).abs()).fold(0.0f32, f32::max);
        assert!(diff < 5e-4, "Grouper HDemix note {i}, error {diff}");
    }
}
