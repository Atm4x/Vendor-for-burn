use crate::reference_loader::{ReferenceLoadError, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Component, Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CachePolicy {
    On,
    Off,
}

#[derive(Clone, Debug)]
pub struct ReferenceEntry<T> {
    pub tokens: T,
    pub text: String,
}

#[derive(Clone, Copy, Debug)]
pub struct ReferenceBytes<'a> {
    pub audio: &'a [u8],
    pub text: &'a str,
}

pub struct ReferenceCache<T> {
    by_audio_hash: HashMap<[u8; 32], T>,
    by_id: HashMap<(std::path::PathBuf, String), Vec<ReferenceEntry<T>>>,
}

impl<T: Clone> Default for ReferenceCache<T> {
    fn default() -> Self {
        Self {
            by_audio_hash: HashMap::new(),
            by_id: HashMap::new(),
        }
    }
}

impl<T: Clone> ReferenceCache<T> {
    pub fn clear(&mut self) {
        self.by_audio_hash.clear();
        self.by_id.clear();
    }

    pub fn entry_counts(&self) -> (usize, usize) {
        (self.by_id.len(), self.by_audio_hash.len())
    }

    pub fn load_by_hash<F>(
        &mut self,
        references: &[ReferenceBytes<'_>],
        policy: CachePolicy,
        mut encode: F,
    ) -> Result<Vec<ReferenceEntry<T>>>
    where
        F: FnMut(&[u8]) -> Result<T>,
    {
        let mut result = Vec::with_capacity(references.len());
        for r in references {
            let digest: [u8; 32] = Sha256::digest(r.audio).into();
            let tokens = if policy == CachePolicy::On {
                if let Some(cached) = self.by_audio_hash.get(&digest) {
                    cached.clone()
                } else {
                    let tokens = encode(r.audio)?;
                    self.by_audio_hash.insert(digest, tokens.clone());
                    tokens
                }
            } else {
                let tokens = encode(r.audio)?;
                self.by_audio_hash.insert(digest, tokens.clone());
                tokens
            };
            result.push(ReferenceEntry {
                tokens,
                text: r.text.to_owned(),
            });
        }
        Ok(result)
    }

    pub fn load_by_id<F>(
        &mut self,
        root: impl AsRef<Path>,
        id: &str,
        policy: CachePolicy,
        mut encode: F,
    ) -> Result<Vec<ReferenceEntry<T>>>
    where
        F: FnMut(&[u8]) -> Result<T>,
    {
        let mut components = Path::new(id).components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            return Err(ReferenceLoadError::Invalid(
                "reference ID must be a single path component",
            ));
        }
        let canonical_root = std::fs::canonicalize(root.as_ref())?;
        let cache_key = (canonical_root.clone(), id.to_owned());
        if policy == CachePolicy::On {
            if let Some(cached) = self.by_id.get(&cache_key) {
                return Ok(cached.clone());
            }
        }
        let reference_dir = canonical_root.join(id);
        let canonical_dir = std::fs::canonicalize(&reference_dir)?;
        if !canonical_dir.starts_with(canonical_root) || !canonical_dir.is_dir() {
            return Err(ReferenceLoadError::Invalid(
                "reference ID escapes reference root",
            ));
        }
        let mut files = Vec::new();
        collect_reference_files(&canonical_dir, &canonical_dir, &mut files)?;
        files.sort();
        let mut result = Vec::with_capacity(files.len());
        for audio_path in files {
            let bytes = std::fs::read(&audio_path)?;
            let tokens = encode(&bytes)?;
            let lab = audio_path.with_extension("lab");
            let text = if lab.is_file() {
                std::fs::read_to_string(&lab)?
            } else {
                lab.to_string_lossy().into_owned()
            };
            result.push(ReferenceEntry { tokens, text });
        }
        self.by_id.insert(cache_key, result.clone());
        Ok(result)
    }
}

fn collect_reference_files(
    root: &Path,
    directory: &Path,
    result: &mut Vec<std::path::PathBuf>,
) -> Result<()> {
    for entry in std::fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_symlink() {
            continue;
        }
        if path.is_dir() {
            collect_reference_files(root, &path, result)?;
        } else if path.is_file() {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if [
                "mp3", "wav", "flac", "ogg", "m4a", "wma", "aac", "aiff", "aif", "aifc",
            ]
            .contains(&ext.as_str())
            {
                if !std::fs::canonicalize(&path)?.starts_with(root) {
                    return Err(ReferenceLoadError::Invalid(
                        "reference audio escapes reference root",
                    ));
                }
                result.push(path);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn hash_cache_reuses_tokens_but_never_stales_text() {
        let mut cache = ReferenceCache::<u32>::default();
        let encoded = Cell::new(0u32);
        let mut make = |_bytes: &[u8]| {
            encoded.set(encoded.get() + 1);
            Ok(encoded.get())
        };
        let refs = [
            ReferenceBytes {
                audio: b"a",
                text: "first",
            },
            ReferenceBytes {
                audio: b"b",
                text: "second",
            },
        ];
        let result = cache
            .load_by_hash(&refs, CachePolicy::On, &mut make)
            .unwrap();
        assert_eq!(
            result.iter().map(|r| r.tokens).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(cache.entry_counts(), (0, 2));
        let updated = [
            ReferenceBytes {
                audio: b"a",
                text: "new caption",
            },
            ReferenceBytes {
                audio: b"b",
                text: "second",
            },
        ];
        let result = cache
            .load_by_hash(&updated, CachePolicy::On, &mut make)
            .unwrap();
        assert_eq!(result[0].tokens, 1);
        assert_eq!(result[0].text, "new caption");
        assert_eq!(encoded.get(), 2);
        let result = cache
            .load_by_hash(&updated, CachePolicy::Off, &mut make)
            .unwrap();
        assert_eq!(result[0].tokens, 3);
        assert_eq!(encoded.get(), 4);
        cache.clear();
        assert_eq!(cache.entry_counts(), (0, 0));
    }

    #[test]
    fn invalid_reference_ids_are_rejected() {
        let mut cache = ReferenceCache::<Vec<u8>>::default();
        for id in ["", "..", "../outside", "nested/../outside", "/tmp", "."] {
            assert!(
                cache
                    .load_by_id("/tmp", id, CachePolicy::On, |b| Ok(b.to_vec()))
                    .is_err()
            );
        }
    }

    #[test]
    fn id_cache_loads_lab_and_respects_policy() {
        let root = std::env::temp_dir().join(format!(
            "fish-ref-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("id")
        ));
        let path = root.join("voice");
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("sample.wav"), b"encoded audio bytes").unwrap();
        std::fs::write(path.join("sample.lab"), "some text").unwrap();
        let mut cache = ReferenceCache::<usize>::default();
        let calls = Cell::new(0);
        let mut encode = |bytes: &[u8]| {
            calls.set(calls.get() + 1);
            Ok(bytes.len())
        };
        let first = cache
            .load_by_id(&root, "voice", CachePolicy::On, &mut encode)
            .unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].text, "some text");
        assert_eq!(calls.get(), 1);
        std::fs::write(path.join("sample.lab"), "changed").unwrap();
        let cached = cache
            .load_by_id(&root, "voice", CachePolicy::On, &mut encode)
            .unwrap();
        assert_eq!(cached[0].text, "some text");
        assert_eq!(calls.get(), 1);
        let uncached = cache
            .load_by_id(&root, "voice", CachePolicy::Off, &mut encode)
            .unwrap();
        assert_eq!(uncached[0].text, "changed");
        assert_eq!(calls.get(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }
}
