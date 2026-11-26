use burn::tensor::Device;
use dettato_rs::evidence::{features::EVIDENCE_FEATURES, model::EvidenceNet, repair::RepairInput};
use serde_json::Value;

#[test]
#[ignore = "released Evidence checkpoint required; set DETTATO_WEIGHTS"]
fn evidence_forward_selected_logits_match_pytorch() {
    let root =
        std::env::var("DETTATO_WEIGHTS").expect("DETTATO_WEIGHTS must point to released weights");
    let model = EvidenceNet::from_directory(root, &Device::default()).unwrap();
    let notes = [
        RepairInput {
            onset: 0.25,
            offset: 0.65,
            pitch: 69,
            velocity: 96,
            original_index: 0,
        },
        RepairInput {
            onset: 0.5,
            offset: 0.6,
            pitch: 70,
            velocity: 80,
            original_index: 1,
        },
    ];
    let evidence: [[f32; EVIDENCE_FEATURES]; 2] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (j as f32 * 0.0157 + i as f32 * 0.2).cos() * 0.01)
    });
    let pred = model.infer(&notes, &evidence, 0.0).unwrap();
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/evidence_forward_reference.json")).unwrap();
    let mut worst = 0.0f64;
    let mut check = |name: &str, i: usize, j: usize, got: f32, want: f64| {
        let error = (f64::from(got) - want).abs();
        worst = worst.max(error);
        assert!(
            error < 0.001,
            "Evidence {name} note={i} index={j}: Burn={got}, PyTorch={want}, error={error}"
        );
    };
    for i in 0..2 {
        let note = &pred.notes[i];
        check(
            "keep",
            i,
            0,
            note.keep,
            fixture["keep"][0][i].as_f64().unwrap(),
        );
        for j in 0..16 {
            check(
                "don",
                i,
                j,
                note.onset[j],
                fixture["heads"]["don"]["rows"][i][j].as_f64().unwrap(),
            );
            check(
                "dur",
                i,
                j,
                note.duration_ratio[j],
                fixture["heads"]["dur"]["rows"][i][j].as_f64().unwrap(),
            );
        }
        for k in 0..2 {
            check(
                "insert",
                i,
                k,
                note.insertion[k].exists,
                fixture["insert_head"][0][i][k].as_f64().unwrap(),
            );
            for j in 0..9 {
                check(
                    "insert_pitch",
                    i,
                    k * 9 + j,
                    note.insertion[k].pitch[j],
                    fixture["insert_pitch"][0][i][k][j].as_f64().unwrap(),
                );
            }
        }
    }
    println!("Evidence selected logits maximum |Burn - PyTorch|: {worst:.9}");
}
