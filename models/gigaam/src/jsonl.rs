//! Line-delimited JSON transport shared by the preloaded CLI and the worker.
use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::io::{self, BufRead, Write};

pub(crate) fn run<F>(mut dispatch: F) -> Result<()>
where
    F: FnMut(Value) -> Result<(Value, bool)>,
{
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line.context("reading JSONL request")?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed = serde_json::from_str::<Value>(&line);
        let id = parsed
            .as_ref()
            .ok()
            .and_then(|value| value.get("id"))
            .cloned()
            .unwrap_or(Value::Null);
        let result = parsed.map_err(Into::into).and_then(&mut dispatch);
        let (response, stop) = match result {
            Ok((value, stop)) => (json!({"id": id, "ok": true, "result": value}), stop),
            Err(error) => (
                json!({"id": id, "ok": false, "error": format!("{error:#}")}),
                false,
            ),
        };
        serde_json::to_writer(&mut output, &response)?;
        output.write_all(b"\n")?;
        output.flush()?;
        if stop {
            break;
        }
    }
    Ok(())
}
