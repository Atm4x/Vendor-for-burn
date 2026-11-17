#![cfg(feature = "tensorrt-mock")]

use fish_speech_core::{
    config::FishModelConfig,
    graph_ir::{Layout, ScalarType, TensorType, ValueId},
    transformer_ir::TransformerIrFactory,
    trtx_blueprint::{Binding, TrtBlueprint, TrtPrimitive, TrtStep},
};
use fish_speech_vendor::trtx_lowering::{LoweringError, preflight_blueprint};

fn step(
    name: &str,
    primitive: TrtPrimitive,
    args: &[Binding],
    result: Binding,
    shape: &[usize],
) -> TrtStep {
    TrtStep {
        name: name.into(),
        primitive,
        arguments: args.to_vec(),
        result,
        output_type: TensorType::new(shape.to_vec(), ScalarType::F32, Layout::Plain),
    }
}

#[test]
fn preflight_valid_connected_graph() {
    let x = Binding::Value(ValueId(0));
    let w = Binding::Value(ValueId(1));
    let y = Binding::Value(ValueId(2));
    let sigmoid = Binding::Temp(0);
    let z = Binding::Value(ValueId(3));
    let plan = TrtBlueprint {
        steps: vec![
            step("input", TrtPrimitive::Input, &[], x, &[1, 4]),
            step("weight", TrtPrimitive::ConstantWeight, &[], w, &[4, 4]),
            step(
                "linear",
                TrtPrimitive::MatrixMultiplyTransposeRhs,
                &[x, w],
                y,
                &[1, 4],
            ),
            step("gate", TrtPrimitive::Sigmoid, &[y], sigmoid, &[1, 4]),
            step(
                "activation",
                TrtPrimitive::ElementwiseProduct,
                &[y, sigmoid],
                z,
                &[1, 4],
            ),
        ],
        outputs: vec![z],
        deferred: Default::default(),
        weight_count: 1,
    };
    let summary = preflight_blueprint(&plan).expect("preflight");
    assert_eq!(summary.network_inputs, 2);
    assert_eq!(summary.unbound_weights, 1);
    assert_eq!(summary.layers, 3);
    assert_eq!(summary.network_outputs, 1);
}

#[test]
fn rejects_deferred_operation_without_network() {
    let x = Binding::Value(ValueId(0));
    let y = Binding::Value(ValueId(1));
    let plan = TrtBlueprint {
        steps: vec![
            step("input", TrtPrimitive::Input, &[], x, &[1, 4]),
            step(
                "unsupported",
                TrtPrimitive::SliceLastSequenceToken,
                &[x],
                y,
                &[1, 4],
            ),
        ],
        outputs: vec![y],
        deferred: Default::default(),
        weight_count: 0,
    };
    assert!(matches!(
        preflight_blueprint(&plan),
        Err(LoweringError::DeferredOperations(_))
    ));
}

#[test]
fn real_slow_graph_reports_unimplemented_backend_ops() {
    let cfg = FishModelConfig::synthetic_dual_ar_fixture();
    let graph = TransformerIrFactory::new(&cfg)
        .unwrap()
        .slow_prefill(2)
        .unwrap();
    let plan = TrtBlueprint::from_graph(&graph).unwrap();
    let err = preflight_blueprint(&plan).unwrap_err();
    let LoweringError::DeferredOperations(reasons) = err else {
        panic!("expected unsupported operations");
    };
    assert!(reasons.iter().any(|reason| reason.contains("RMSNorm")));
    assert!(reasons.iter().any(|reason| reason.contains("GQA")));
    assert!(reasons.iter().any(|reason| reason.contains("KV")));
}
