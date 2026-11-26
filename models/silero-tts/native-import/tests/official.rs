use silero_package::EnglishCheckpoint;
use pytorch_reader::DType;
use std::path::PathBuf;

#[test]
#[ignore = "requires official v3_en.pt; set SILERO_V3_EN_CHECKPOINT"]
fn official_v3_en_all_tensors() {
    let path = PathBuf::from(std::env::var_os("SILERO_V3_EN_CHECKPOINT").expect("checkpoint path"));
    let package = EnglishCheckpoint::open(&path).unwrap();
    assert_eq!(package.tensors().len(), 363);
    assert_eq!(package.symbols.chars().count(), 39);
    assert_eq!(package.speakers["en_0"], 0);
    assert_eq!(package.encode("Hello, world!").unwrap(), [2,20,17,24,24,27,7,3,35,27,30,24,16,4,1]);
    assert!(package.encode("Привет!").is_err());
    let mut total = 0;
    for (name, t) in package.tensors() {
        assert_eq!(t.dtype(), DType::F32, "{name}");
        let bytes = t.read().unwrap();
        assert_eq!(bytes.len(), t.byte_len(), "{name}");
        assert!(bytes.chunks_exact(4).all(|c| f32::from_ne_bytes(c.try_into().unwrap()).is_finite()), "{name}");
        total += bytes.len();
    }
    assert_eq!(total, 80_199_960);
}

#[test]
#[ignore = "requires official v5_ru.pt; set SILERO_V5_RU_CHECKPOINT"]
fn official_v5_ru_all_components() {
    let path = PathBuf::from(std::env::var_os("SILERO_V5_RU_CHECKPOINT").expect("checkpoint path"));
    let package = silero_package::RussianCheckpoint::open(&path).unwrap();
    assert_eq!(package.config.symbols.chars().count(),47);
    assert_eq!(package.config.speakers["xenia"],4);
    assert_eq!(package.ngrams.len(),126523);
    assert_eq!(package.tokenizer.vocab.len(),83830);
    assert_eq!(package.homographs.len(),1924);
    for (weights,count) in [(&package.acoustic,311),(&package.accentor,19),(&package.homosolver,63),(&package.pqmf,2)] {
        assert_eq!(weights.tensors().len(),count);
        for t in weights.tensors().values() { assert_eq!(t.read().unwrap().len(),t.byte_len()); }
    }
    let values=package.homosolver.read_f32("bert.embeddings.word_embeddings.weight").unwrap();
    assert_eq!(values.len(),83830*312);
    assert!(values.iter().all(|v|v.is_finite()));
}
