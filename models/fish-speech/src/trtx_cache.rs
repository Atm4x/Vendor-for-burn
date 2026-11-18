use std::{
    collections::{BTreeMap, VecDeque},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use fish_speech_core::trtx_blueprint::TrtBlueprint;
use sha2::{Digest, Sha256};

use crate::trtx_engine::EngineError;

const CACHE_FORMAT: &[u8; 12] = b"FISHTRTX0002";
const HEADER_LEN: usize = CACHE_FORMAT.len() + 32 + 8 + 32;
const MAX_ENGINE_BYTES: usize = 16 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineIdentity {
    pub model_sha256: String,
    pub graph_sha256: String,
    pub gpu_identity: String,
    pub tensorrt_version: String,
    pub driver_version: String,
    pub precision: String,
    pub shape_profile: String,
    pub build_flags: String,
}

impl EngineIdentity {
    pub fn for_blueprint(
        blueprint: &TrtBlueprint,
        model_sha256: &str,
        gpu_identity: &str,
        tensorrt_version: &str,
        driver_version: &str,
        precision: &str,
        shape_profile: &str,
        build_flags: &str,
    ) -> Result<Self, EngineError> {
        let mut fingerprint = Sha256::new();
        fingerprint.update(b"fish-speech-trtx-blueprint-v2");
        fingerprint.update(format!("{blueprint:?}").as_bytes());
        let result = Self {
            model_sha256: model_sha256.to_owned(),
            graph_sha256: format!("{:x}", fingerprint.finalize()),
            gpu_identity: gpu_identity.to_owned(),
            tensorrt_version: tensorrt_version.to_owned(),
            driver_version: driver_version.to_owned(),
            precision: precision.to_owned(),
            shape_profile: shape_profile.to_owned(),
            build_flags: build_flags.to_owned(),
        };
        result.key()?;
        Ok(result)
    }

    pub fn validate_blueprint(&self, blueprint: &TrtBlueprint) -> Result<(), EngineError> {
        let mut digest = Sha256::new();
        digest.update(b"fish-speech-trtx-blueprint-v2");
        digest.update(format!("{blueprint:?}").as_bytes());
        if self.graph_sha256 != format!("{:x}", digest.finalize()) {
            return Err(EngineError::InvalidPlan(
                "cache identity does not match the requested TensorRT blueprint".into(),
            ));
        }
        self.key()?;
        Ok(())
    }

    pub fn key(&self) -> Result<String, EngineError> {
        for digest in [&self.model_sha256, &self.graph_sha256] {
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err(EngineError::InvalidPlan(
                    "cache identity requires 64-character SHA-256 fingerprints".into(),
                ));
            }
        }
        let mut hasher = Sha256::new();
        hasher.update(b"fish-speech-trtx-engine-cache-v2");
        for field in [
            &self.model_sha256,
            &self.graph_sha256,
            &self.gpu_identity,
            &self.tensorrt_version,
            &self.driver_version,
            &self.precision,
            &self.shape_profile,
            &self.build_flags,
        ] {
            if field.is_empty() || field.len() > 256 {
                return Err(EngineError::InvalidPlan(
                    "invalid engine cache identity field".into(),
                ));
            }
            hasher.update((field.len() as u32).to_le_bytes());
            hasher.update(field.as_bytes());
        }
        Ok(format!("{:x}", hasher.finalize()))
    }
}

#[derive(Clone, Debug)]
pub struct CachedEngine {
    pub key: String,
    pub bytes: Arc<[u8]>,
}

struct CacheState {
    entries: BTreeMap<String, Arc<CachedEngine>>,
    eviction_order: VecDeque<String>,
    resident_bytes: usize,
}

pub struct EngineRegistry {
    directory: PathBuf,
    max_engine_bytes: usize,
    max_resident_bytes: usize,
    loaded: Mutex<CacheState>,
}

impl EngineRegistry {
    pub fn new(directory: impl AsRef<Path>, max_engine_bytes: usize) -> Result<Self, EngineError> {
        Self::new_with_resident_budget(directory, max_engine_bytes, max_engine_bytes)
    }

    pub fn new_with_resident_budget(
        directory: impl AsRef<Path>,
        max_engine_bytes: usize,
        max_resident_bytes: usize,
    ) -> Result<Self, EngineError> {
        if max_engine_bytes == 0
            || max_engine_bytes > MAX_ENGINE_BYTES
            || max_resident_bytes == 0
            || max_resident_bytes < max_engine_bytes
        {
            return Err(EngineError::InvalidPlan(
                "invalid engine cache size or total resident-memory budget".into(),
            ));
        }
        Ok(Self {
            directory: directory.as_ref().to_owned(),
            max_engine_bytes,
            max_resident_bytes,
            loaded: Mutex::new(CacheState {
                entries: BTreeMap::new(),
                eviction_order: VecDeque::new(),
                resident_bytes: 0,
            }),
        })
    }

    pub fn get_or_build(
        &self,
        identity: &EngineIdentity,
        builder: impl FnOnce() -> Result<Vec<u8>, EngineError>,
    ) -> Result<Arc<CachedEngine>, EngineError> {
        let key = identity.key()?;
        let mut loaded = self
            .loaded
            .lock()
            .map_err(|_| EngineError::InvalidBindings("engine registry lock poisoned".into()))?;
        if let Some(engine) = loaded.entries.get(&key) {
            return Ok(Arc::clone(engine));
        }
        fs::create_dir_all(&self.directory)
            .map_err(|e| EngineError::TensorRt(format!("engine cache directory: {e}")))?;
        let destination = self.directory.join(format!("{key}.rtxplan"));
        let bytes = match read_verified(&destination, &key, self.max_engine_bytes) {
            Ok(Some(bytes)) => bytes,
            Err(error) => {
                return Err(EngineError::TensorRt(format!("engine cache read: {error}")));
            }
            Ok(None) => {
                let bytes = builder()?;
                if bytes.is_empty() || bytes.len() > self.max_engine_bytes {
                    return Err(EngineError::InvalidPlan(
                        "serialized TensorRT engine is empty or exceeds cache budget".into(),
                    ));
                }
                store_atomic(&destination, &key, &bytes)?;
                bytes
            }
        };
        let entry = Arc::new(CachedEngine {
            key: key.clone(),
            bytes: bytes.into(),
        });
        while loaded.resident_bytes.saturating_add(entry.bytes.len()) > self.max_resident_bytes {
            let oldest = loaded.eviction_order.pop_front().ok_or_else(|| {
                EngineError::InvalidPlan("engine resident-memory budget exhausted".into())
            })?;
            if let Some(removed) = loaded.entries.remove(&oldest) {
                loaded.resident_bytes -= removed.bytes.len();
            }
        }
        loaded.resident_bytes += entry.bytes.len();
        loaded.eviction_order.push_back(key.clone());
        loaded.entries.insert(key, Arc::clone(&entry));
        Ok(entry)
    }

    #[cfg(feature = "tensorrt-api")]
    pub fn build_or_load_blueprint(
        &self,
        logger: &trtx::Logger,
        blueprint: &TrtBlueprint,
        identity: &EngineIdentity,
        workspace_bytes: usize,
    ) -> Result<Arc<CachedEngine>, EngineError> {
        identity.validate_blueprint(blueprint)?;
        self.get_or_build(identity, || {
            crate::trtx_engine::build_serialized_blueprint(logger, blueprint, workspace_bytes)
        })
    }

    #[cfg(feature = "tensorrt-api")]
    pub fn with_cached_session<T>(
        &self,
        logger: &trtx::Logger,
        blueprint: &TrtBlueprint,
        identity: &EngineIdentity,
        workspace_bytes: usize,
        submit: impl FnOnce(&mut trtx::ExecutionContext<'_>) -> Result<T, EngineError>,
        synchronize: impl FnOnce() -> Result<(), EngineError>,
    ) -> Result<T, EngineError> {
        let image = self.build_or_load_blueprint(logger, blueprint, identity, workspace_bytes)?;
        crate::trtx_engine::with_engine_session(logger, &image.bytes, submit, synchronize)
    }

    pub fn evict_memory(&self) -> Result<(), EngineError> {
        self.loaded
            .lock()
            .map_err(|_| EngineError::InvalidBindings("engine registry lock poisoned".into()))?
            .clear_all();
        Ok(())
    }

    pub fn resident_bytes(&self) -> Result<usize, EngineError> {
        Ok(self
            .loaded
            .lock()
            .map_err(|_| EngineError::InvalidBindings("engine registry lock poisoned".into()))?
            .resident_bytes)
    }

    pub fn resident_engines(&self) -> Result<usize, EngineError> {
        Ok(self
            .loaded
            .lock()
            .map_err(|_| EngineError::InvalidBindings("engine registry lock poisoned".into()))?
            .entries
            .len())
    }
}

impl CacheState {
    fn clear_all(&mut self) {
        self.entries.clear();
        self.eviction_order.clear();
        self.resident_bytes = 0;
    }
}

fn read_verified(path: &Path, key: &str, max_bytes: usize) -> std::io::Result<Option<Vec<u8>>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let len = file.metadata()?.len();
    if len < HEADER_LEN as u64 || len > (HEADER_LEN + max_bytes) as u64 {
        return Ok(None);
    }
    let mut header = [0u8; HEADER_LEN];
    file.read_exact(&mut header)?;
    let key_bytes = hex_sha256(key).ok_or_else(|| std::io::Error::other("bad cache key"))?;
    if &header[..12] != CACHE_FORMAT || header[12..44] != key_bytes {
        return Ok(None);
    }
    let mut expected_size = [0u8; 8];
    expected_size.copy_from_slice(&header[44..52]);
    let expected_size = u64::from_le_bytes(expected_size);
    if expected_size == 0
        || expected_size > max_bytes as u64
        || expected_size.checked_add(HEADER_LEN as u64) != Some(len)
    {
        return Ok(None);
    }
    let mut bytes = Vec::with_capacity(expected_size as usize);
    file.read_to_end(&mut bytes)?;
    if bytes.len() != expected_size as usize || Sha256::digest(&bytes).as_slice() != &header[52..84]
    {
        return Ok(None);
    }
    Ok(Some(bytes))
}

fn hex_sha256(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    for (i, item) in bytes.iter_mut().enumerate() {
        *item = u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(bytes)
}

fn store_atomic(path: &Path, key: &str, engine: &[u8]) -> Result<(), EngineError> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let suffix = NEXT.fetch_add(1, Ordering::Relaxed);
    let temp = path.with_extension(format!("{}.{}.part", std::process::id(), suffix));
    let result = (|| -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        let digest = hex_sha256(key).ok_or_else(|| std::io::Error::other("bad cache key"))?;
        file.write_all(CACHE_FORMAT)?;
        file.write_all(&digest)?;
        file.write_all(&(engine.len() as u64).to_le_bytes())?;
        file.write_all(&Sha256::digest(engine))?;
        file.write_all(engine)?;
        file.sync_all()?;
        match fs::rename(&temp, path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                fs::remove_file(path)?;
                fs::rename(&temp, path)?;
            }
            Err(error) => return Err(error),
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result.map_err(|e| EngineError::TensorRt(format!("engine cache write: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn sample() -> EngineIdentity {
        EngineIdentity {
            model_sha256: "11".repeat(32),
            graph_sha256: "22".repeat(32),
            gpu_identity: "sm89-rtx4060".into(),
            tensorrt_version: "1.7".into(),
            driver_version: "test-driver".into(),
            precision: "bf16".into(),
            shape_profile: "prefill:128".into(),
            build_flags: "opt0".into(),
        }
    }

    #[test]
    fn cached_engine_survives_registry_restart_and_is_not_rebuilt() {
        let directory = std::env::temp_dir().join(format!(
            "fish-engine-cache-{}-{}",
            std::process::id(),
            NEXT_TEST.fetch_add(1, Ordering::Relaxed)
        ));
        let counter = AtomicUsize::new(0);
        let identity = sample();
        let registry = EngineRegistry::new(&directory, 1024).unwrap();
        let first = registry
            .get_or_build(&identity, || {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(vec![1, 2, 3])
            })
            .unwrap();
        let second = registry
            .get_or_build(&identity, || panic!("duplicate compilation"))
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        drop(registry);
        let registry = EngineRegistry::new(&directory, 1024).unwrap();
        let reloaded = registry
            .get_or_build(&identity, || panic!("unexpected rebuild"))
            .unwrap();
        assert_eq!(&*reloaded.bytes, &[1, 2, 3]);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        fs::remove_dir_all(directory).unwrap();
    }

    static NEXT_TEST: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn corrupt_entry_is_rebuilt_and_key_includes_device_profile_and_model() {
        let directory = std::env::temp_dir().join(format!(
            "fish-engine-corrupt-{}-{}",
            std::process::id(),
            NEXT_TEST.fetch_add(1, Ordering::Relaxed)
        ));
        let initial = sample();
        let mut alternate = initial.clone();
        alternate.gpu_identity.push_str("-different");
        assert_ne!(initial.key().unwrap(), alternate.key().unwrap());
        alternate = initial.clone();
        alternate.model_sha256 = "33".repeat(32);
        assert_ne!(initial.key().unwrap(), alternate.key().unwrap());
        let registry = EngineRegistry::new(&directory, 1024).unwrap();
        registry.get_or_build(&initial, || Ok(vec![5, 5])).unwrap();
        registry.evict_memory().unwrap();
        let key = initial.key().unwrap();
        fs::write(directory.join(format!("{key}.rtxplan")), b"corrupt").unwrap();
        let rebuilt = registry.get_or_build(&initial, || Ok(vec![9, 9])).unwrap();
        assert_eq!(&*rebuilt.bytes, &[9, 9]);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn bounded_resident_cache_evicts_without_invalidating_external_lease() {
        let directory = std::env::temp_dir().join(format!(
            "fish-engine-resident-{}-{}",
            std::process::id(),
            NEXT_TEST.fetch_add(1, Ordering::Relaxed)
        ));
        let registry = EngineRegistry::new_with_resident_budget(&directory, 4, 6).unwrap();
        let first_id = sample();
        let mut second_id = first_id.clone();
        second_id.shape_profile = "decode:t1".into();
        let first = registry
            .get_or_build(&first_id, || Ok(vec![1, 2, 3, 4]))
            .unwrap();
        registry
            .get_or_build(&second_id, || Ok(vec![5, 6, 7, 8]))
            .unwrap();
        assert_eq!(registry.resident_engines().unwrap(), 1);
        assert_eq!(registry.resident_bytes().unwrap(), 4);
        assert_eq!(&*first.bytes, &[1, 2, 3, 4]);
        registry
            .get_or_build(&first_id, || panic!("should load from disk"))
            .unwrap();
        assert_eq!(registry.resident_engines().unwrap(), 1);
        assert_eq!(registry.resident_bytes().unwrap(), 4);
        registry.evict_memory().unwrap();
        assert_eq!(registry.resident_bytes().unwrap(), 0);
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn forged_overflowing_length_in_cache_header_is_rebuilt_safely() {
        let directory = std::env::temp_dir().join(format!(
            "fish-engine-overflow-{}-{}",
            std::process::id(),
            NEXT_TEST.fetch_add(1, Ordering::Relaxed)
        ));
        let registry = EngineRegistry::new(&directory, 128).unwrap();
        let identity = sample();
        let key = identity.key().unwrap();
        fs::create_dir_all(&directory).unwrap();
        let mut corrupt = Vec::new();
        corrupt.extend_from_slice(CACHE_FORMAT);
        corrupt.extend_from_slice(&hex_sha256(&key).unwrap());
        corrupt.extend_from_slice(&u64::MAX.to_le_bytes());
        corrupt.extend_from_slice(&[0; 32]);
        fs::write(directory.join(format!("{key}.rtxplan")), corrupt).unwrap();
        let rebuilt = registry.get_or_build(&identity, || Ok(vec![42])).unwrap();
        assert_eq!(&*rebuilt.bytes, &[42]);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn invalid_fingerprints_empty_builds_and_budget_are_rejected() {
        let mut identity = sample();
        identity.model_sha256 = "wrong".into();
        assert!(identity.key().is_err());
        let identity = sample();
        assert!(EngineRegistry::new("unused", 0).is_err());
        let directory = std::env::temp_dir().join(format!(
            "fish-engine-budget-{}-{}",
            std::process::id(),
            NEXT_TEST.fetch_add(1, Ordering::Relaxed)
        ));
        let registry = EngineRegistry::new(&directory, 2).unwrap();
        assert!(
            registry
                .get_or_build(&identity, || Ok(vec![1, 2, 3]))
                .is_err()
        );
        assert_eq!(registry.resident_engines().unwrap(), 0);
        fs::remove_dir_all(directory).unwrap();
    }
}
