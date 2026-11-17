//! Model-name resolution from Python gigaam.__init__ and an opt-in HTTPS cache.
//! MD5 checksums are upstream artifact identifiers, not cryptographic authentication.
use anyhow::{Context, Result, bail, ensure};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};
#[cfg(feature = "model-download")]
use std::{io::Write, time::Duration};

const ROOT_URL: &str = "https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM";
const MODEL_HASHES: &[(&str, &str)] = &[
    ("emo", "7ce76f9535cb254488985057c0d33006"),
    ("v1_ctc", "f027f199e590a391d015aeede2e66174"),
    ("v1_rnnt", "02c758999bcdc6afcb2087ef256d47ef"),
    ("v1_ssl", "dc7f7b231f7f91c4968dc21910e7b396"),
    ("v2_ctc", "e00f59cb5d39624fb30d1786044795bf"),
    ("v2_rnnt", "547460139acfebd842323f59ed54ab54"),
    ("v2_ssl", "cd4cf819c8191a07b9d7edcad111668e"),
    ("v3_ctc", "73413e7be9c6a5935827bfab5c0dd678"),
    ("v3_rnnt", "0fd2c9a1ff66abd8d32a3a07f7592815"),
    ("v3_e2e_ctc", "367074d6498f426d960b25f49531cf68"),
    ("v3_e2e_rnnt", "2730de7545ac43ad256485a462b0a27a"),
    ("v3_ssl", "70cbf5ed7303a0ed242ddb257e9dc6a6"),
    ("multilingual_ctc", "5379d887c53ccd9cb95981e2a1832720"),
    ("multilingual_ssl", "af54fed7a0337eeae7c4a25b2f8779c8"),
    ("multilingual_large_ctc", "79a9adde50dd7f35bbf70927cb6557d0"),
    ("multilingual_large_ssl", "2ef65a2ca413f6e1f99a4df0e86c1cee"),
];

#[derive(Clone, Debug)]
pub struct CacheOptions {
    pub directory: PathBuf,
    pub download_if_missing: bool,
    pub validate_md5: bool,
}

impl Default for CacheOptions {
    fn default() -> Self {
        let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"));
        Self {
            directory: home
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cache")
                .join("gigaam"),
            download_if_missing: false,
            validate_md5: true,
        }
    }
}

pub fn known_models() -> &'static [(&'static str, &'static str)] {
    MODEL_HASHES
}

/// Resolve a directory, checkpoint path, or one of the Python model names.
/// Network requests are disabled unless `download_if_missing` is true.
pub fn resolve_model_path(spec: &str, options: &CacheOptions) -> Result<PathBuf> {
    let expanded =
        if let Some(suffix) = spec.strip_prefix("~/").or_else(|| spec.strip_prefix("~\\")) {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(PathBuf::from)
                .context("cannot expand home in model path")?
                .join(suffix)
        } else {
            PathBuf::from(spec)
        };
    if expanded.is_file() || expanded.is_dir() {
        return Ok(expanded);
    }
    let canonical = crate::ModelKind::canonical_model_name(spec);
    let md5 = MODEL_HASHES
        .iter()
        .find(|(name, _)| *name == canonical)
        .map(|(_, md5)| *md5);
    let Some(md5) = md5 else {
        bail!("model path does not exist or unsupported model name: {spec}");
    };
    let ckpt = options.directory.join(format!("{canonical}.ckpt"));
    ensure_model_file(
        &ckpt,
        &format!("{ROOT_URL}/{canonical}.ckpt"),
        Some(md5),
        options,
    )?;
    if canonical == "v1_rnnt" || canonical.contains("e2e") {
        let filename = format!("{canonical}_tokenizer.model");
        ensure_model_file(
            &options.directory.join(&filename),
            &format!("{ROOT_URL}/{filename}"),
            None,
            options,
        )?;
    }
    Ok(ckpt)
}

fn ensure_model_file(
    path: &Path,
    url: &str,
    expected_md5: Option<&str>,
    options: &CacheOptions,
) -> Result<()> {
    if !path.is_file() {
        ensure!(
            options.download_if_missing,
            "model artifact missing: {} (enable download_if_missing or provide a local path)",
            path.display()
        );
        fs::create_dir_all(path.parent().context("artifact has no parent directory")?)?;
        let mut error = None;
        for _ in 0..3 {
            match download_atomic(url, path) {
                Ok(()) => {
                    error = None;
                    break;
                }
                Err(err) => error = Some(err),
            }
        }
        if let Some(err) = error {
            return Err(err);
        }
    }
    if options.validate_md5 {
        if let Some(expected) = expected_md5 {
            let actual = md5_file(path)?;
            ensure!(
                actual == expected,
                "GigaAM checkpoint MD5 mismatch for {}: expected {expected}, received {actual}",
                path.display()
            );
        }
    }
    Ok(())
}

#[cfg(feature = "model-download")]
fn download_atomic(url: &str, path: &Path) -> Result<()> {
    let temp = path.with_extension(format!(
        "{}.part",
        path.extension().and_then(|x| x.to_str()).unwrap_or("tmp")
    ));
    let attempt = (|| -> Result<()> {
        let response = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(300))
            .build()
            .get(url)
            .call()
            .with_context(|| format!("downloading {url}"))?;
        let mut reader = response.into_reader();
        let mut target = File::create(&temp)?;
        std::io::copy(&mut reader, &mut target)?;
        target.flush()?;
        target.sync_all()?;
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if attempt.is_err() {
        let _ = fs::remove_file(&temp);
    }
    attempt
}

#[cfg(not(feature = "model-download"))]
fn download_atomic(_url: &str, _path: &Path) -> Result<()> {
    bail!("model download requires compiling with --features model-download")
}

const SHIFT: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];
const TABLE: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

struct Md5 {
    state: [u32; 4],
    buffer: [u8; 64],
    buffered: usize,
    total: u64,
}
impl Default for Md5 {
    fn default() -> Self {
        Self {
            state: [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476],
            buffer: [0; 64],
            buffered: 0,
            total: 0,
        }
    }
}
impl Md5 {
    fn block(&mut self, data: &[u8]) {
        let mut words = [0u32; 16];
        for (i, chunk) in data.chunks_exact(4).enumerate() {
            words[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        let [mut a, mut b, mut c, mut d] = self.state;
        for i in 0..64 {
            let (f, g) = match i {
                0..=15 => ((b & c) | (!b & d), i),
                16..=31 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                32..=47 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let next = a
                .wrapping_add(f)
                .wrapping_add(TABLE[i])
                .wrapping_add(words[g])
                .rotate_left(SHIFT[i]);
            a = d;
            d = c;
            c = b;
            b = b.wrapping_add(next);
        }
        for (dest, value) in self.state.iter_mut().zip([a, b, c, d]) {
            *dest = dest.wrapping_add(value);
        }
    }
    fn update(&mut self, mut data: &[u8]) {
        self.total = self.total.wrapping_add(data.len() as u64);
        if self.buffered > 0 {
            let take = (64 - self.buffered).min(data.len());
            self.buffer[self.buffered..self.buffered + take].copy_from_slice(&data[..take]);
            self.buffered += take;
            data = &data[take..];
            if self.buffered == 64 {
                let block = self.buffer;
                self.block(&block);
                self.buffered = 0;
            }
        }
        for chunk in data.chunks_exact(64) {
            self.block(chunk);
        }
        let rest = data.len() % 64;
        if rest > 0 {
            self.buffer[..rest].copy_from_slice(&data[data.len() - rest..]);
            self.buffered = rest;
        }
    }
    fn finish(mut self) -> String {
        let bit_len = self.total.wrapping_mul(8);
        let used = self.buffered;
        let padding = if used < 56 { 56 - used } else { 120 - used };
        let mut bytes = vec![0x80];
        bytes.resize(padding, 0);
        self.update(&bytes);
        self.update(&bit_len.to_le_bytes());
        self.state
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .map(|b| format!("{b:02x}"))
            .collect()
    }
}
fn md5_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut md5 = Md5::default();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        md5.update(&buffer[..bytes]);
    }
    Ok(md5.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn md5_matches_standard_vectors_and_chunked() {
        for (data, expected) in [
            ("", "d41d8cd98f00b204e9800998ecf8427e"),
            ("a", "0cc175b9c0f1b6a831c399e269772661"),
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            ("message digest", "f96b697d7cb7938d525a2f31aaf161d0"),
        ] {
            let mut h = Md5::default();
            for chunk in data.as_bytes().chunks(2) {
                h.update(chunk);
            }
            assert_eq!(h.finish(), expected);
        }
    }
    #[test]
    fn aliases_and_cache_do_not_access_network_without_permission() -> Result<()> {
        assert_eq!(crate::ModelKind::canonical_model_name("ctc"), "v3_ctc");
        let options = CacheOptions {
            directory: std::env::temp_dir().join("gigaam-nonexistent-model-cache-test"),
            download_if_missing: false,
            validate_md5: true,
        };
        assert!(
            resolve_model_path("ctc", &options)
                .unwrap_err()
                .to_string()
                .contains("missing")
        );
        assert!(resolve_model_path("bad-model", &options).is_err());
        Ok(())
    }
}
