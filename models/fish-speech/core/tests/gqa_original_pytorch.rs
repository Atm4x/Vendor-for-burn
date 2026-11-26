use std::collections::BTreeMap;

use fish_speech_core::ar_attention::{causal_gqa_cpu, GqaShape};
use fish_speech_core::cpu_ir::{NamedTensors, TensorData};
use fish_speech_core::graph::Stage;
use fish_speech_core::graph_ir::{Graph, Layout, Op, ScalarType, TensorType};
use fish_speech_core::trtx_blueprint::TrtBlueprint;

fn goldens() -> BTreeMap<String, Vec<f32>> {
    include_str!("../fixtures/gqa_pytorch.txt")
        .lines()
        .map(|line| {
            let (key, rest) = line.split_once(':').unwrap();
            let vals = rest.split(',').map(|v| v.parse::<f32>().unwrap()).collect();
            (key.to_string(), vals)
        })
        .collect()
}

#[test]
fn original_pytorch_sdpa_matches_rust_grouped_attention() {
    let goldens = goldens();
    for n in 0..3 {
        let sizes: Vec<usize> = goldens[&format!("shape{n}")]
            .iter()
            .map(|&v| v as usize)
            .collect();
        let [b, qh, kh, qlen, kvlen, dim]: [usize; 6] = sizes.try_into().unwrap();
        let shape = GqaShape {
            batch: b,
            query_heads: qh,
            kv_heads: kh,
            query_len: qlen,
            kv_len: kvlen,
            head_dim: dim,
            first_position: kvlen - qlen,
        };
        let q = &goldens[&format!("q{n}")];
        let k = &goldens[&format!("k{n}")];
        let v = &goldens[&format!("v{n}")];
        let expected = &goldens[&format!("output{n}")];
        let actual = causal_gqa_cpu(q, k, v, shape).unwrap();
        assert_eq!(actual.len(), expected.len());
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs < 1.0e-5, "scenario {n}, max={max_abs}");

        let mut graph = Graph::new(Stage::SlowDecode, None, None);
        let make = |graph: &mut Graph, name: &str, heads: usize, len: usize, dtype| {
            graph
                .input(
                    name,
                    TensorType::new([b, heads, len, dim], dtype, Layout::HeadsBhtd),
                )
                .unwrap()
        };
        let qid = make(&mut graph, "q", qh, qlen, ScalarType::F32);
        let kid = make(&mut graph, "k", kh, kvlen, ScalarType::BF16);
        let vid = make(&mut graph, "v", kh, kvlen, ScalarType::BF16);
        let out = graph
            .emit(
                "attention",
                Op::GqaAttention {
                    position: kvlen - qlen,
                    q_heads: qh,
                    kv_heads: kh,
                },
                &[qid, kid, vid],
            )
            .unwrap();
        graph.output(out).unwrap();
        let blueprint = TrtBlueprint::from_graph(&graph).unwrap();
        blueprint.require_no_deferred_ops().unwrap();
        let mut bindings = NamedTensors::new();
        bindings.insert("q".into(), TensorData::F32(q.clone()));
        bindings.insert("k".into(), TensorData::F32(k.clone()));
        bindings.insert("v".into(), TensorData::F32(v.clone()));
        let inferred = blueprint.execute_cpu_reference(&bindings).unwrap();
        let TensorData::F32(inferred) = &inferred["attention"] else {
            panic!("F32 output expected")
        };
        assert_eq!(actual, *inferred);
    }
}
