use crate::{DType, PickleValue as V, Tensor, WeightSet, dictionary, ids, require_tensor, string};
use anyhow::{Context, Result, bail, ensure};
use std::collections::{BTreeMap, HashMap};

pub struct QuantizedProjection {
    pub input: usize,
    pub output: usize,
    pub weights: Vec<i8>,
    pub scale: f32,
    pub zero: i32,
    pub bias: Vec<f32>,
}

pub struct RecurrentAccentCheckpoint {
    pub graphemes: BTreeMap<String, usize>,
    pub exceptions: BTreeMap<String, usize>,
    pub vowels: String,
    pub stress_threshold: f32,
    pub yo_threshold: f32,
    pub embedding: Vec<f32>,
    pub layers: Vec<[QuantizedProjection; 2]>,
    pub head: WeightSet,
}

fn list<'a>(v: Option<&'a V>, label: &str) -> Result<&'a [V]> {
    match v {
        Some(V::List(xs)) => Ok(xs),
        _ => bail!("invalid {label}"),
    }
}
fn integer(v: Option<&V>, label: &str) -> Result<i64> {
    match v {
        Some(V::Int(i)) => Ok(*i),
        _ => bail!("invalid {label}"),
    }
}
fn number(v: Option<&V>, label: &str) -> Result<f32> {
    match v {
        Some(V::Float(x)) if (*x as f32).is_finite() => Ok(*x as f32),
        _ => bail!("invalid {label}"),
    }
}
fn load_projection(
    tensors: &HashMap<String, Tensor>,
    state: &V,
    prefix: &str,
    input: usize,
) -> Result<QuantizedProjection> {
    let obj = dictionary(state, "packed linear")?;
    ensure!(
        string(obj.get("class"), "packed linear class")?
            == "__torch__.torch.classes.quantized.LinearPackedParamsBase",
        "invalid packed linear class"
    );
    let state = list(obj.get("state"), "packed linear state")?;
    ensure!(state.len() == 2, "invalid packed linear state arity");
    let quant = dictionary(&state[0], "quantizer")?;
    let scale = number(quant.get("scale"), "quantizer scale")?;
    let zero = integer(quant.get("zero_point"), "quantizer zero")?;
    ensure!(
        scale > 0.0 && (-128..=127).contains(&zero),
        "invalid quantizer"
    );
    let weight_name = format!("{prefix}.state.0.weight");
    let bias_name = format!("{prefix}.state.1");
    require_tensor(tensors, &weight_name, DType::I8, &[512, input])?;
    require_tensor(tensors, &bias_name, DType::F32, &[512])?;
    let weights = tensors[&weight_name]
        .read()?
        .into_iter()
        .map(|v| v as i8)
        .collect();
    let bias: Vec<_> = tensors[&bias_name]
        .read()?
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
        .collect();
    ensure!(
        bias.iter().all(|v| v.is_finite()),
        "nonfinite recurrent bias"
    );
    Ok(QuantizedProjection {
        input,
        output: 512,
        weights,
        scale,
        zero: zero as i32,
        bias,
    })
}

pub(crate) fn load_recurrent_accent(
    root: &HashMap<String, V>,
    tensors: &HashMap<String, Tensor>,
) -> Result<RecurrentAccentCheckpoint> {
    let accent = dictionary(
        root.get("accentor").context("missing recurrent accentor")?,
        "recurrent accentor",
    )?;
    let encoder = dictionary(
        accent.get("encoder").context("missing recurrent encoder")?,
        "recurrent encoder",
    )?;
    for (name, val) in [
        ("hidden_size", 128),
        ("input_size", 300),
        ("num_layers", 4),
        ("version", 2),
    ] {
        ensure!(
            integer(encoder.get(name), name)? == val,
            "unsupported recurrent encoder {name}"
        );
    }
    for name in ["bidirectional", "batch_first", "bias"] {
        ensure!(
            matches!(encoder.get(name), Some(V::Bool(true))),
            "unsupported recurrent encoder {name}"
        );
    }
    ensure!(
        number(encoder.get("dropout"), "recurrent dropout")? == 0.0,
        "unsupported recurrent dropout"
    );
    let cells = dictionary(
        encoder
            .get("_all_weight_values")
            .context("missing packed cells")?,
        "packed cells",
    )?;
    ensure!(
        (0..8).all(|i| cells.contains_key(&i.to_string())),
        "expected eight packed cells: {:?}",
        cells.keys()
    );
    let mut layers = Vec::with_capacity(8);
    for i in 0..8 {
        let cell = dictionary(
            cells.get(&i.to_string()).context("missing packed cell")?,
            "packed cell",
        )?;
        let param = dictionary(
            cell.get("param").context("missing packed cell parameter")?,
            "packed cell parameter",
        )?;
        ensure!(
            string(param.get("class"), "packed cell class")?
                == "__torch__.torch.classes.rnn.CellParamsBase",
            "invalid packed cell class"
        );
        let state = list(param.get("state"), "packed cell state")?;
        ensure!(
            state.len() == 5 && string(state.first(), "packed cell mode")? == "quantized_dynamic",
            "unsupported packed cell mode"
        );
        ensure!(
            matches!(list(state.get(3), "packed cell version")?, [V::Int(1)]),
            "unsupported packed cell version"
        );
        let projections = list(state.get(4), "packed cell projections")?;
        ensure!(projections.len() == 2, "invalid packed projection count");
        let prefix = format!("accentor.encoder._all_weight_values.{i}.param.state.4");
        let wi = load_projection(
            tensors,
            &projections[0],
            &format!("{prefix}.0"),
            if i < 2 { 300 } else { 256 },
        )?;
        let wh = load_projection(tensors, &projections[1], &format!("{prefix}.1"), 128)?;
        layers.push([wi, wh]);
    }
    let graphemes = ids(root.get("grapheme2idx"), "grapheme dictionary")?;
    ensure!(
        graphemes.len() == 37 && graphemes.values().all(|&i| i < 37),
        "invalid recurrent grapheme dictionary"
    );
    let exceptions = ids(root.get("exceptions"), "stress exceptions")?;
    ensure!(
        exceptions.len() <= 100000
            && exceptions
                .iter()
                .all(|(w, &p)| w.is_ascii() && p <= w.len()),
        "invalid recurrent exceptions"
    );
    let vowels = string(root.get("vowels"), "vowels")?.to_owned();
    ensure!(
        vowels.is_ascii()
            && string(root.get("stress_token"), "stress token")? == "+"
            && string(root.get("yo_token"), "yo token")? == "1",
        "unsupported recurrent accent alphabet"
    );
    let stress_threshold = number(root.get("stress_trs"), "stress threshold")?;
    let yo_threshold = number(root.get("yo_trs"), "yo threshold")?;
    ensure!(
        (0.0..=1.0).contains(&stress_threshold) && (0.0..=1.0).contains(&yo_threshold),
        "invalid stress thresholds"
    );
    require_tensor(tensors, "accentor.embedding.weight", DType::F32, &[37, 300])?;
    require_tensor(
        tensors,
        "accentor.attention.attention_weights",
        DType::F32,
        &[256],
    )?;
    for name in ["stress_clf", "yo_clf"] {
        for (idx, input, output) in [(0, 256, 128), (1, 128, 64), (2, 64, 11)] {
            require_tensor(
                tensors,
                &format!("accentor.{name}.{idx}.weight"),
                DType::F32,
                &[output, input],
            )?;
            require_tensor(
                tensors,
                &format!("accentor.{name}.{idx}.bias"),
                DType::F32,
                &[output],
            )?;
        }
    }
    let head = WeightSet::new(
        tensors
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix("accentor.")
                    .filter(|s| !s.starts_with("encoder."))
                    .map(|s| (s.to_owned(), v.clone()))
            })
            .collect(),
    )?;
    let embedding = head.read_f32("embedding.weight")?;
    Ok(RecurrentAccentCheckpoint {
        graphemes,
        exceptions,
        vowels,
        stress_threshold,
        yo_threshold,
        embedding,
        layers,
        head,
    })
}
