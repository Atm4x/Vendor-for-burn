use anyhow::{Context, Result, ensure};
use std::{fs::File, io::BufReader, path::Path};

/// Identify the serialized format from archive entries, not extension or model ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArchiveLayout { TtsPackage, TorchScript }

impl ArchiveLayout {
    pub fn inspect(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref()).context("open Silero checkpoint")?;
        let archive = zip::ZipArchive::new(BufReader::new(file)).context("invalid Silero ZIP container")?;
        let names = archive.file_names().collect::<zip::result::ZipResult<Vec<_>>>()?;
        let packages: Vec<_> = names.iter().filter_map(|n| n.strip_suffix("tts_models/model"))
            .filter(|root| root.is_empty() || root.ends_with('/')).collect();
        ensure!(packages.len() <= 1, "ambiguous Silero package root");
        if !packages.is_empty() { return Ok(Self::TtsPackage); }
        let scripts = names.iter().filter_map(|n| n.strip_suffix("data.pkl"))
            .filter(|root| root.is_empty() || root.ends_with('/'))
            .filter(|root| names.iter().any(|name| **name == format!("{root}constants.pkl")))
            .count();
        ensure!(scripts == 1, "unsupported Silero archive layout (no TTS package or unique TorchScript root)");
        Ok(Self::TorchScript)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn distinguishes_layouts_without_filename_hints() {
        let path = std::env::temp_dir().join(format!("silero-layout-{}.bin", std::process::id()));
        for (names, expected) in [
            (vec!["root/tts_models/model", "root/.data/ts_code/0/data.pkl", "root/.data/ts_code/0/constants.pkl"], Some(ArchiveLayout::TtsPackage)),
            (vec!["root/data.pkl", "root/constants.pkl"], Some(ArchiveLayout::TorchScript)),
            (vec!["root/data.pkl"], None),
            (vec!["a/tts_models/model", "b/tts_models/model"], None),
        ] {
            let mut zip = zip::ZipWriter::new(File::create(&path).unwrap());
            for name in names {
                zip.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
                zip.write_all(b"test").unwrap();
            }
            zip.finish().unwrap();
            assert_eq!(ArchiveLayout::inspect(&path).ok(), expected);
        }
        std::fs::remove_file(path).unwrap();
    }
}
