//! Restricted, data-only PyTorch pickle reader for OmegaConf checkpoint metadata.
//! The parser does not invoke Python constructors or deserialize tensor storages.
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Number, Value};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    path::Path,
};

const MAX_PICKLE_BYTES: u64 = 100 * 1024 * 1024;
const MAX_NODES: usize = 4_000_000;
const MARK: usize = usize::MAX;

#[derive(Clone, Debug)]
enum Node {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Sequence(Vec<usize>),
    Mapping(Vec<(usize, usize)>),
    Global(String),
    Object { class: String, state: Option<usize> },
    Opaque,
}

struct Pickle<'a> {
    data: &'a [u8],
    offset: usize,
    nodes: Vec<Node>,
    stack: Vec<usize>,
    memo: HashMap<u32, usize>,
    next_memo: u32,
}

impl<'a> Pickle<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            offset: 0,
            nodes: Vec::new(),
            stack: Vec::new(),
            memo: HashMap::new(),
            next_memo: 0,
        }
    }

    fn bytes(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .context("pickle offset overflow")?;
        ensure!(end <= self.data.len(), "truncated checkpoint pickle");
        let output = &self.data[self.offset..end];
        self.offset = end;
        Ok(output)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.bytes(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into()?))
    }
    fn line(&mut self) -> Result<&'a [u8]> {
        let rel = self.data[self.offset..]
            .iter()
            .position(|b| *b == b'\n')
            .context("unterminated pickle line")?;
        let line = self.bytes(rel)?;
        self.bytes(1)?;
        Ok(line)
    }
    fn text(&mut self, len: usize) -> Result<String> {
        Ok(std::str::from_utf8(self.bytes(len)?)
            .context("pickle string is not UTF-8")?
            .to_owned())
    }
    fn put(&mut self, node: Node) -> Result<usize> {
        ensure!(self.nodes.len() < MAX_NODES, "too many pickle objects");
        let id = self.nodes.len();
        self.nodes.push(node);
        self.stack.push(id);
        Ok(id)
    }
    fn push(&mut self, id: usize) -> Result<()> {
        ensure!(self.stack.len() < MAX_NODES, "pickle nesting is too large");
        self.stack.push(id);
        Ok(())
    }
    fn top(&self) -> Result<usize> {
        self.stack
            .last()
            .copied()
            .filter(|id| *id < self.nodes.len())
            .context("invalid pickle stack top")
    }
    fn pop(&mut self) -> Result<usize> {
        self.stack
            .pop()
            .filter(|id| *id != MARK)
            .context("malformed pickle stack")
    }
    fn mark_values(&mut self) -> Result<Vec<usize>> {
        let index = self
            .stack
            .iter()
            .rposition(|id| *id == MARK)
            .context("pickle MARK not found")?;
        let values = self.stack.split_off(index + 1);
        self.stack.pop();
        Ok(values)
    }
    fn memorize(&mut self, id: u32) -> Result<()> {
        let current = *self
            .stack
            .last()
            .filter(|id| **id != MARK)
            .context("pickle memo on empty stack")?;
        self.memo.insert(id, current);
        self.next_memo = self.next_memo.max(id.saturating_add(1));
        Ok(())
    }
    fn unpickle(&mut self) -> Result<usize> {
        while self.offset < self.data.len() {
            let opcode = self.u8()?;
            match opcode {
                b'.' => return self.pop(),
                0x80 => {
                    ensure!(self.u8()? <= 5, "pickle protocol > 5 unsupported");
                }
                0x95 => {
                    self.u64()?;
                }
                b'(' => self.push(MARK)?,
                b'N' => {
                    self.put(Node::Null)?;
                }
                0x88 => {
                    self.put(Node::Bool(true))?;
                }
                0x89 => {
                    self.put(Node::Bool(false))?;
                }
                b'K' => {
                    let v = self.u8()?;
                    self.put(Node::Integer(i64::from(v)))?;
                }
                b'M' => {
                    let v = self.u16()?;
                    self.put(Node::Integer(i64::from(v)))?;
                }
                b'J' => {
                    let v = self.u32()?;
                    self.put(Node::Integer(i32::from_le_bytes(v.to_le_bytes()) as i64))?;
                }
                b'G' => {
                    let v = f64::from_be_bytes(self.bytes(8)?.try_into()?);
                    self.put(Node::Float(v))?;
                }
                b'F' => {
                    let v = std::str::from_utf8(self.line()?)?.parse::<f64>()?;
                    self.put(Node::Float(v))?;
                }
                b'I' | b'L' => {
                    let raw = std::str::from_utf8(self.line()?)?;
                    let raw = raw.trim_end_matches('L');
                    if raw == "00" || raw == "False" {
                        self.put(Node::Bool(false))?;
                    } else if raw == "01" || raw == "True" {
                        self.put(Node::Bool(true))?;
                    } else {
                        self.put(Node::Integer(raw.parse::<i64>()?))?;
                    }
                }
                0x8a | 0x8b => {
                    let len = if opcode == 0x8a {
                        self.u8()? as usize
                    } else {
                        self.u32()? as usize
                    };
                    let data = self.bytes(len)?;
                    let mut bytes = [0u8; 8];
                    let value = if len <= 8 {
                        if len > 0 && data[len - 1] & 0x80 != 0 {
                            bytes.fill(0xff);
                        }
                        bytes[..len].copy_from_slice(data);
                        Node::Integer(i64::from_le_bytes(bytes))
                    } else {
                        Node::Opaque
                    };
                    self.put(value)?;
                }
                b'X' => {
                    let len = self.u32()? as usize;
                    let value = self.text(len)?;
                    self.put(Node::String(value))?;
                }
                0x8c => {
                    let len = self.u8()? as usize;
                    let value = self.text(len)?;
                    self.put(Node::String(value))?;
                }
                0x8d => {
                    let len = usize::try_from(self.u64()?)?;
                    let value = self.text(len)?;
                    self.put(Node::String(value))?;
                }
                b'V' => {
                    let value = std::str::from_utf8(self.line()?)?.to_owned();
                    self.put(Node::String(value))?;
                }
                b'U' | b'C' => {
                    let len = self.u8()? as usize;
                    self.bytes(len)?;
                    self.put(Node::Opaque)?;
                }
                b'T' | b'B' => {
                    let len = self.u32()? as usize;
                    self.bytes(len)?;
                    self.put(Node::Opaque)?;
                }
                0x8e | 0x96 => {
                    let len = usize::try_from(self.u64()?)?;
                    self.bytes(len)?;
                    self.put(Node::Opaque)?;
                }
                b'c' => {
                    let module = std::str::from_utf8(self.line()?)?.to_owned();
                    let name = std::str::from_utf8(self.line()?)?.to_owned();
                    self.put(Node::Global(format!("{module}.{name}")))?;
                }
                0x93 => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (Node::String(name), Node::String(module)) =
                        (&self.nodes[name], &self.nodes[module])
                    else {
                        bail!("invalid STACK_GLOBAL")
                    };
                    self.put(Node::Global(format!("{module}.{name}")))?;
                }
                b')' | b']' | b'}' | 0x8f => {
                    let node = if opcode == b'}' {
                        Node::Mapping(Vec::new())
                    } else {
                        Node::Sequence(Vec::new())
                    };
                    self.put(node)?;
                }
                b't' | b'l' | b'd' | 0x91 => {
                    let values = self.mark_values()?;
                    if opcode == b'd' {
                        ensure!(values.len() % 2 == 0, "uneven pickle dict");
                        self.put(Node::Mapping(
                            values
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .map(|x| (x[0], x[1]))
                                .collect(),
                        ))?;
                    } else {
                        self.put(Node::Sequence(values))?;
                    }
                }
                0x85..=0x87 => {
                    let len = (opcode - 0x84) as usize;
                    let mut values = (0..len).map(|_| self.pop()).collect::<Result<Vec<_>>>()?;
                    values.reverse();
                    self.put(Node::Sequence(values))?;
                }
                b'a' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    let Node::Sequence(values) = &mut self.nodes[k] else {
                        bail!("APPEND requires a list")
                    };
                    values.push(v);
                    self.push(k)?;
                }
                b'e' | 0x90 => {
                    let values = self.mark_values()?;
                    let target = self.top()?;
                    let Node::Sequence(list) = &mut self.nodes[target] else {
                        bail!("APPENDS requires a list")
                    };
                    list.extend(values);
                }
                b's' => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    let target = self.top()?;
                    let Node::Mapping(map) = &mut self.nodes[target] else {
                        bail!("SETITEM requires a dictionary")
                    };
                    map.push((key, value));
                }
                b'u' => {
                    let values = self.mark_values()?;
                    ensure!(values.len() % 2 == 0, "uneven SETITEMS");
                    let target = self.top()?;
                    let Node::Mapping(map) = &mut self.nodes[target] else {
                        bail!("SETITEMS requires a dictionary")
                    };
                    map.extend(values.as_chunks::<2>().0.iter().map(|x| (x[0], x[1])));
                }
                0x81 | 0x92 | b'R' => {
                    let arg = self.pop()?;
                    if opcode == 0x92 {
                        self.pop()?;
                    }
                    let callable = self.pop()?;
                    let cls = match &self.nodes[callable] {
                        Node::Global(s) => s.as_str(),
                        Node::Object { class, .. } => class.as_str(),
                        _ => "",
                    };
                    let node = if opcode == b'R'
                        && matches!(cls, "collections.OrderedDict" | "collections.defaultdict")
                    {
                        Node::Mapping(Vec::new())
                    } else if opcode == b'R' {
                        Node::Opaque
                    } else {
                        Node::Object {
                            class: cls.to_owned(),
                            state: None,
                        }
                    };
                    let _ = arg;
                    self.put(node)?;
                }
                b'b' => {
                    let state = self.pop()?;
                    let target = self.pop()?;
                    if let Node::Object {
                        state: obj_state, ..
                    } = &mut self.nodes[target]
                    {
                        *obj_state = Some(state)
                    }
                    self.push(target)?;
                }
                b'Q' => {
                    self.pop()?;
                    self.put(Node::Opaque)?;
                }
                b'P' => {
                    self.line()?;
                    self.put(Node::Opaque)?;
                }
                b'h' => {
                    let i = self.u8()? as u32;
                    let v = *self.memo.get(&i).context("invalid pickle memo ID")?;
                    self.push(v)?;
                }
                b'j' => {
                    let i = self.u32()?;
                    let v = *self.memo.get(&i).context("invalid pickle memo ID")?;
                    self.push(v)?;
                }
                b'q' => {
                    let i = self.u8()? as u32;
                    self.memorize(i)?;
                }
                b'r' => {
                    let i = self.u32()?;
                    self.memorize(i)?;
                }
                0x94 => {
                    let id = self.next_memo;
                    self.memorize(id)?;
                }
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    self.mark_values()?;
                }
                b'2' => {
                    let id = self.top()?;
                    self.push(id)?;
                }
                _ => bail!(
                    "unsupported pickle opcode 0x{opcode:02x} at offset {}",
                    self.offset - 1
                ),
            }
            ensure!(
                self.stack.len() < MAX_NODES,
                "pickle stack exceeds safe limit"
            );
        }
        bail!("pickle missing STOP")
    }

    fn get(&self, id: usize, key: &str) -> Option<usize> {
        let Node::Mapping(items) = self.nodes.get(id)? else {
            return None;
        };
        items.iter().rev().find_map(|&(k, v)| match &self.nodes[k] {
            Node::String(s) if s == key => Some(v),
            _ => None,
        })
    }

    fn json_value(&self, id: usize, depth: usize, visited: &mut HashSet<usize>) -> Result<Value> {
        ensure!(depth < 100, "OmegaConf nesting is too deep");
        ensure!(visited.insert(id), "cycle in checkpoint configuration");
        let result = match &self.nodes[id] {
            Node::Null => Value::Null,
            Node::Bool(v) => Value::Bool(*v),
            Node::Integer(v) => Value::Number(Number::from(*v)),
            Node::Float(v) => {
                Value::Number(Number::from_f64(*v).context("nonfinite checkpoint float")?)
            }
            Node::String(v) => Value::String(v.clone()),
            Node::Sequence(values) => Value::Array(
                values
                    .iter()
                    .map(|id| self.json_value(*id, depth + 1, visited))
                    .collect::<Result<Vec<_>>>()?,
            ),
            Node::Mapping(values) => {
                let mut out = Map::new();
                for &(k, v) in values {
                    let name = match &self.nodes[k] {
                        Node::String(s) => s.clone(),
                        Node::Integer(n) => n.to_string(),
                        _ => bail!("unsupported key in configuration dictionary"),
                    };
                    out.insert(name, self.json_value(v, depth + 1, visited)?);
                }
                Value::Object(out)
            }
            Node::Object { class, state } => {
                let state = state.context("checkpoint object without state")?;
                if class == "omegaconf.dictconfig.DictConfig"
                    || class == "omegaconf.listconfig.ListConfig"
                {
                    let content = self
                        .get(state, "_content")
                        .context("OmegaConf container lacks _content")?;
                    self.json_value(content, depth + 1, visited)?
                } else if class.starts_with("omegaconf.nodes.") {
                    let val = self
                        .get(state, "_val")
                        .context("OmegaConf scalar lacks _val")?;
                    self.json_value(val, depth + 1, visited)?
                } else {
                    bail!("unsupported checkpoint configuration object: {class}")
                }
            }
            Node::Global(class) => bail!("unsupported global configuration value: {class}"),
            Node::Opaque => bail!("opaque value in checkpoint configuration"),
        };
        visited.remove(&id);
        Ok(result)
    }
}

/// Read a built-in JSON object or OmegaConf `DictConfig` from PyTorch's `data.pkl`.
/// Only scalar/list/map checkpoint metadata is returned. Pickle callables are never executed.
pub struct CheckpointConfig {
    pub config: Value,
    pub base_model_name: Option<String>,
    pub is_finetuned: bool,
}

pub fn read_checkpoint_metadata(path: &Path) -> Result<CheckpointConfig> {
    let file =
        File::open(path).with_context(|| format!("opening checkpoint {}", path.display()))?;
    let mut zip =
        zip::ZipArchive::new(file).context("checkpoint must be a modern PyTorch ZIP archive")?;
    let mut data = Vec::new();
    let mut found = false;
    for i in 0..zip.len() {
        let entry = zip.by_index(i)?;
        let name = entry.name()?;
        if !name.ends_with("/data.pkl") && name != "data.pkl" {
            continue;
        }
        ensure!(!found, "multiple data.pkl entries in checkpoint");
        ensure!(
            entry.size() <= MAX_PICKLE_BYTES,
            "checkpoint metadata exceeds maximum size"
        );
        entry.take(MAX_PICKLE_BYTES + 1).read_to_end(&mut data)?;
        ensure!(
            data.len() as u64 <= MAX_PICKLE_BYTES,
            "checkpoint metadata exceeds maximum size"
        );
        found = true;
    }
    ensure!(found, "checkpoint has no data.pkl");
    let mut pickle = Pickle::new(&data);
    let root = pickle.unpickle()?;
    let hp = pickle.get(root, "hyper_parameters");
    let direct_cfg = pickle.get(root, "cfg");
    let cfg = direct_cfg.or_else(|| hp.and_then(|h| pickle.get(h, "model_cfg")));
    let base_model_name = hp
        .and_then(|h| pickle.get(h, "model_name"))
        .and_then(|id| match &pickle.nodes[id] {
            Node::String(s) => Some(s.clone()),
            _ => None,
        });
    Ok(CheckpointConfig {
        config: cfg.map_or(Ok(Value::Null), |id| {
            pickle.json_value(id, 0, &mut HashSet::new())
        })?,
        base_model_name,
        is_finetuned: hp.is_some() && direct_cfg.is_none(),
    })
}

pub fn read_checkpoint_config(path: &Path) -> Result<Value> {
    let cfg = read_checkpoint_metadata(path)?.config;
    ensure!(!cfg.is_null(), "checkpoint has no cfg or hyper_parameters.model_cfg");
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn root_dict_pickle() {
        let bytes =
            b"\x80\x02}\x71\x00X\x03\x00\x00\x00cfgq\x01}q\x02X\x03\x00\x00\x00fooq\x03K\x07ss.";
        let mut p = Pickle::new(bytes);
        let root = p.unpickle().unwrap();
        let cfg = p.get(root, "cfg").unwrap();
        assert_eq!(p.json_value(cfg, 0, &mut HashSet::new()).unwrap()["foo"], 7);
    }
    #[test]
    fn malformed_pickle_never_panics() {
        for seed in 0..128u64 {
            let mut state = seed + 1;
            let mut bytes = vec![0u8; 64];
            for v in &mut bytes {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                *v = state as u8;
            }
            for len in [0usize, 1, 3, 7, 16, 64] {
                let result = std::panic::catch_unwind(|| {
                    let mut reader = Pickle::new(&bytes[..len]);
                    reader.unpickle()
                });
                assert!(
                    result.is_ok(),
                    "pickle parser panicked on seed={seed}, len={len}"
                );
            }
        }
    }

    #[test]
    fn parses_genuine_omegaconf_pickle_fixture() {
        let path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/omegaconf_checkpoint.ckpt"
        ));
        let cfg = read_checkpoint_config(path).unwrap();
        assert_eq!(cfg["encoder"]["n_layers"], 2);
        assert_eq!(cfg["preprocessor"]["sample_rate"], 16000);
        assert_eq!(cfg["decoding"]["vocabulary"][0], "а");
    }
}
