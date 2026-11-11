use serde::Serialize;
use std::{env, process::ExitCode, time::Instant};
use tts_with_rvc_rs::faiss_index::IvfFlatIndex;

#[derive(Serialize)]
struct Case {
    rows: usize,
    index_rate: f32,
    serial_ms: f64,
    bulk_ms: f64,
    exact_equal: bool,
    max_abs_error: f32,
    max_delta_from_input: f32,
    output_digest: String,
    sample_neighbors: Vec<(usize, Vec<(usize, u32)>)>,
}

fn execute() -> Result<(), String> {
    let index_path = env::args().nth(1).ok_or("usage: bench_faiss_real_stage55 <path-to-index>")?;
    let index = IvfFlatIndex::open(&index_path)?;
    let dim = index.dimension();
    let mut cases = Vec::new();
    for rows in [1, 7, 31, 32, 33] {
        let mut input = Vec::with_capacity(rows * dim);
        for row in 0..rows {
            let id = row.wrapping_mul(7_919).wrapping_add(19) % index.len();
            let source = index.reconstruct(id).ok_or("missing test vector")?;
            input.extend(source.iter().enumerate().map(|(channel, &value)| {
                value + (((row * 17 + channel * 13) % 11) as f32 - 5.0) * 0.0007
            }));
        }
        for rate in [0.0_f32, 0.75] {
            let mut serial = input.clone();
            let start_serial = Instant::now();
            for query in serial.chunks_exact_mut(dim) {
                index.blend_features(query, rate)?;
            }
            let serial_ms = start_serial.elapsed().as_secs_f64() * 1000.0;
            let mut bulk = input.clone();
            let start_bulk = Instant::now();
            index.blend_features(&mut bulk, rate)?;
            let bulk_ms = start_bulk.elapsed().as_secs_f64() * 1000.0;
            let max_abs_error = serial.iter().zip(&bulk)
                .map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
            let max_delta_from_input = input.iter().zip(&bulk)
                .map(|(&a, &b)| (a - b).abs()).fold(0.0f32, f32::max);
            let exact_equal = serial == bulk;
            if !exact_equal {
                return Err(format!("FAISS serial/bulk mismatch: rows={rows}, rate={rate}, max={max_abs_error}"));
            }
            let output_digest = format!("{:016x}", serial.iter().flat_map(|v|v.to_bits().to_le_bytes())
                .fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3)));
            let sample_neighbors = if rate > 0.0 {
                [1usize, 2, 4].into_iter().map(|probe| {
                    let matches = index.search_with_nprobe(&input[..dim], 8, probe).unwrap();
                    (probe, matches.iter().map(|n| (n.id, n.squared_distance.to_bits())).collect())
                }).collect()
            } else { Vec::new() };
            cases.push(Case { rows, index_rate: rate, serial_ms, bulk_ms, exact_equal,
                max_abs_error, max_delta_from_input, output_digest, sample_neighbors });
        }
    }
    println!("{}", serde_json::to_string_pretty(&serde_json::json!({
        "stage":55, "backend":"native-cpu", "scope":"real-IVFFlat-L2-index",
        "dimension":dim, "vectors":index.len(), "nlist":index.nlist(), "nprobe":index.nprobe(),
        "cases":cases, "all_passed":true
    })).map_err(|e|e.to_string())?);
    Ok(())
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => { eprintln!("{error}"); ExitCode::FAILURE }
    }
}
