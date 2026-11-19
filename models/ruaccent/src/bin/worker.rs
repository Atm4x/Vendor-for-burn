use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
};

use ruaccent_rs::RuAccent;
use serde::Deserialize;
use serde_json::{json, Value};

const MAX_REQUEST_BYTES: usize = 1024 * 1024;

#[derive(Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
enum Request {
    Capabilities {
        request_id: String,
    },
    Load {
        request_id: String,
        model_dir: PathBuf,
        #[serde(default = "default_model")]
        omograph_model: String,
    },
    Process {
        request_id: String,
        text: String,
        #[serde(default)]
        skip_regex: Option<String>,
    },
    Status {
        request_id: String,
    },
    Unload {
        request_id: String,
    },
    Shutdown {
        request_id: String,
    },
}

fn default_model() -> String {
    "tiny".to_owned()
}

impl Request {
    fn id(&self) -> &str {
        match self {
            Self::Capabilities { request_id }
            | Self::Load { request_id, .. }
            | Self::Process { request_id, .. }
            | Self::Status { request_id }
            | Self::Unload { request_id }
            | Self::Shutdown { request_id } => request_id,
        }
    }
}

#[derive(Default)]
struct Worker {
    model: Option<RuAccent>,
    processed: u64,
}

impl Worker {
    fn execute(&mut self, req: Request) -> Result<(Value, bool), String> {
        match req {
            Request::Capabilities { .. } => Ok((
                json!({
                    "protocol_version": 1,
                    "model": "ruaccent",
                    "native_rust": true,
                    "model_pack_required": true,
                    "onnx_runtime": "load-dynamic",
                    "stress_notation": "+",
                    "commands": ["capabilities", "load", "process", "status", "unload", "shutdown"]
                }),
                false,
            )),
            Request::Load {
                model_dir,
                omograph_model,
                ..
            } => {
                if self.model.is_some() {
                    return Err("RUAccent already loaded; unload first".into());
                }
                let resolved = model_dir
                    .canonicalize()
                    .map_err(|e| format!("RUAccent model directory is inaccessible: {e}"))?;
                let model = RuAccent::load_with_omograph(&resolved, &omograph_model)
                    .map_err(|e| format!("RUAccent model initialization failed: {e}"))?;
                self.model = Some(model);
                self.processed = 0;
                Ok((self.status(), false))
            }
            Request::Process {
                text, skip_regex, ..
            } => {
                if text.chars().count() > 16_384 {
                    return Err("RUAccent text exceeds 16384 characters".into());
                }
                let model = self.model.as_mut().ok_or("RUAccent model is not loaded")?;
                let stressed = match skip_regex {
                    Some(regex) => {
                        regex::Regex::new(&regex)
                            .map_err(|e| format!("Invalid skip_regex: {e}"))?;
                        model.process_all_with_skip(&text, &regex)
                    }
                    None => model.process_all(&text),
                };
                self.processed += 1;
                Ok((json!({ "text": stressed, "stress_notation": "+" }), false))
            }
            Request::Status { .. } => Ok((self.status(), false)),
            Request::Unload { .. } => {
                self.model = None;
                self.processed = 0;
                Ok((self.status(), false))
            }
            Request::Shutdown { .. } => {
                self.model = None;
                Ok((json!({"state": "shutdown"}), true))
            }
        }
    }

    fn status(&self) -> Value {
        json!({"state": if self.model.is_some() {"ready"} else {"unloaded"},
            "successful_requests": self.processed})
    }
}

fn read_frame<R: BufRead>(reader: &mut R, frame: &mut Vec<u8>) -> io::Result<Option<bool>> {
    frame.clear();
    let mut oversized = false;
    let mut consumed = false;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return if consumed {
                Ok(Some(!oversized))
            } else {
                Ok(None)
            };
        }
        let newline = available.iter().position(|byte| *byte == b'\n');
        let count = newline.map(|idx| idx + 1).unwrap_or(available.len());
        if !oversized {
            if frame.len().saturating_add(count) > MAX_REQUEST_BYTES {
                oversized = true;
                frame.clear();
            } else {
                frame.extend_from_slice(&available[..count]);
            }
        }
        reader.consume(count);
        consumed = true;
        if newline.is_some() {
            return Ok(Some(!oversized));
        }
    }
}

fn serve<R: BufRead, W: Write>(mut reader: R, mut writer: W) -> io::Result<()> {
    let mut worker = Worker::default();
    let mut frame = Vec::new();
    loop {
        let Some(allowed) = read_frame(&mut reader, &mut frame)? else {
            break;
        };
        let (request_id, response, done) = if !allowed {
            (None, Err("Request exceeds 1 MiB".to_owned()), false)
        } else {
            match serde_json::from_slice::<Request>(&frame) {
                Ok(req) => {
                    let id = req.id().to_owned();
                    let result = worker.execute(req);
                    let done = result.as_ref().map(|(_, stop)| *stop).unwrap_or(false);
                    (Some(id), result.map(|(data, _)| data), done)
                }
                Err(error) => (None, Err(format!("Invalid request: {error}")), false),
            }
        };
        let answer = match response {
            Ok(data) => json!({"request_id": request_id, "ok": true, "data": data}),
            Err(error) => json!({"request_id": request_id, "ok": false, "error": error}),
        };
        serde_json::to_writer(&mut writer, &answer)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        if done {
            break;
        };
    }
    Ok(())
}

fn main() -> io::Result<()> {
    serve(io::stdin().lock(), io::stdout().lock())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn protocol_handles_status_errors_and_shutdown_without_models() {
        let requests = concat!(
            "{\"command\":\"capabilities\",\"request_id\":\"a\"}\n",
            "{\"command\":\"process\",\"request_id\":\"b\",\"text\":\"Привет!\"}\n",
            "{\"command\":\"status\",\"request_id\":\"c\"}\n",
            "{\"command\":\"shutdown\",\"request_id\":\"d\"}\n"
        );
        let mut output = Vec::new();
        serve(Cursor::new(requests), &mut output).unwrap();
        let replies = String::from_utf8(output).unwrap();
        let values: Vec<Value> = replies
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(values.len(), 4);
        assert_eq!(values[0]["data"]["model"], "ruaccent");
        assert_eq!(values[1]["ok"], false);
        assert_eq!(values[2]["data"]["state"], "unloaded");
        assert_eq!(values[3]["data"]["state"], "shutdown");
    }

    #[test]
    fn oversized_frame_is_consumed_and_connection_recovers() {
        let mut data = String::new();
        data.push_str(&"z".repeat(MAX_REQUEST_BYTES));
        data.push_str("\n{\"command\":\"status\",\"request_id\":\"after\"}\n");
        let mut output = Vec::new();
        serve(Cursor::new(data), &mut output).unwrap();
        let values: Vec<Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0]["ok"], false);
        assert_eq!(values[1]["request_id"], "after");
        assert_eq!(values[1]["ok"], true);
    }
}
