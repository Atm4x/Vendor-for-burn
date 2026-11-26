use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufRead, Read, Write},
    path::{Path, PathBuf},
};
use uuid::Uuid;

pub const MAX_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    version: u32,
    id: Uuid,
    command: String,
    payload: Value,
}

pub fn serve(mut dispatch: impl FnMut(&str, Value) -> Result<Value, String>) -> Result<(), String> {
    let mut input = io::stdin().lock();
    let mut output = io::stdout().lock();
    let mut frame = Vec::new();
    loop {
        frame.clear();
        let count = input
            .by_ref()
            .take((MAX_FRAME_BYTES + 1) as u64)
            .read_until(b'\n', &mut frame)
            .map_err(|error| error.to_string())?;
        if count == 0 {
            return Ok(());
        }
        if count > MAX_FRAME_BYTES || frame.last() != Some(&b'\n') {
            return Err("invalid worker request frame".into());
        }
        let request: Request = serde_json::from_slice(&frame).map_err(|error| error.to_string())?;
        let result = if request.version != 1 || !request.payload.is_object() {
            Err("expected protocol version 1 and an object payload".into())
        } else {
            dispatch(&request.command, request.payload)
        };
        let stop = request.command == "shutdown" && result.is_ok();
        let response = match result {
            Ok(result) => {
                json!({"version":1,"id":request.id,"ok":true,"result":result,"error":null})
            }
            Err(error) => {
                json!({"version":1,"id":request.id,"ok":false,"result":null,"error":error})
            }
        };
        serde_json::to_writer(&mut output, &response).map_err(|error| error.to_string())?;
        output
            .write_all(b"\n")
            .and_then(|()| output.flush())
            .map_err(|error| error.to_string())?;
        if stop {
            return Ok(());
        }
    }
}

pub fn absolute_path(path: &Path, directory: bool) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("expected an absolute path: {}", path.display()));
    }
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if metadata.is_symlink()
        || (directory && !metadata.is_dir())
        || (!directory && !metadata.is_file())
    {
        return Err(format!("unexpected file type: {}", path.display()));
    }
    path.canonicalize().map_err(|error| error.to_string())
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Snapshot {
    pub file_name: String,
    pub size_bytes: u64,
    pub sha256: String,
}

impl Snapshot {
    pub fn read(&self, root: &Path) -> Result<Vec<u8>, String> {
        if self.file_name.is_empty()
            || self.file_name.len() > 128
            || self.file_name.starts_with('.')
            || !self
                .file_name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
            || !(1..=MAX_SOURCE_BYTES).contains(&self.size_bytes)
            || self.sha256.len() != 64
        {
            return Err("invalid input snapshot descriptor".into());
        }
        let path = absolute_path(&root.join(&self.file_name), false)?;
        if path.parent() != Some(root) {
            return Err("input snapshot is outside the exchange directory".into());
        }
        let mut bytes = Vec::new();
        File::open(path)
            .map_err(|error| error.to_string())?
            .take(self.size_bytes + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        if bytes.len() as u64 != self.size_bytes || digest(&bytes) != self.sha256 {
            return Err("input snapshot size or SHA-256 does not match".into());
        }
        Ok(bytes)
    }

    pub fn write(root: &Path, extension: &str, bytes: &[u8]) -> Result<Self, String> {
        if extension.is_empty() || !extension.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
            return Err("invalid output file extension".into());
        }
        let file_name = format!("{}.{extension}", Uuid::new_v4());
        let path = root.join(&file_name);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| error.to_string())?;
        if let Err(error) = file.write_all(bytes).and_then(|()| file.flush()) {
            drop(file);
            let _ = fs::remove_file(path);
            return Err(error.to_string());
        }
        Ok(Self {
            file_name,
            size_bytes: bytes.len() as u64,
            sha256: digest(bytes),
        })
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
