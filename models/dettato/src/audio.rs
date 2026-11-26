use std::{fs::File, io::Cursor, path::Path};
use symphonia::core::{
    codecs::audio::AudioDecoderOptions,
    errors::Error,
    formats::{FormatOptions, TrackType, probe::Hint},
    io::MediaSourceStream,
    meta::MetadataOptions,
};

pub struct MonoAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Decode audio once at its source sample rate, downmixing channels by their
/// arithmetic mean (equivalent to the original Dettato loader).
pub fn decode(path: impl AsRef<Path>) -> Result<MonoAudio, String> {
    let path=path.as_ref();
    let source=MediaSourceStream::new(Box::new(File::open(path).map_err(|e|e.to_string())?),Default::default());
    decode_source(source, path.extension().and_then(|extension| extension.to_str()))
}

pub fn decode_snapshot(bytes: Vec<u8>, extension: &str) -> Result<MonoAudio, String> {
    let source = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    decode_source(source, Some(extension))
}

fn decode_source(source: MediaSourceStream, extension: Option<&str>) -> Result<MonoAudio, String> {
    let mut hint=Hint::new();
    if let Some(ext)=extension{hint.with_extension(ext);}
    let mut format=symphonia::default::get_probe().probe(
        &hint,source,FormatOptions::default(),MetadataOptions::default(),
    ).map_err(|e|format!("probing audio: {e}"))?;
    let track=format.default_track(TrackType::Audio).ok_or("no audio track")?;
    let id=track.id;
    let params=track.codec_params.as_ref().ok_or("missing codec parameters")?
        .audio().ok_or("source track is not audio")?;
    let mut decoder=symphonia::default::get_codecs()
        .make_audio_decoder(params,&AudioDecoderOptions::default())
        .map_err(|e|format!("creating audio decoder: {e}"))?;
    let mut rate=None;
    let mut samples=Vec::new();
    loop {
        let packet=match format.next_packet() {
            Ok(Some(packet))=>packet,
            Ok(None)=>break,
            Err(Error::ResetRequired)=>return Err("audio stream format changed".into()),
            Err(e)=>return Err(format!("reading audio packet: {e}")),
        };
        if packet.track_id!=id{continue;}
        let decoded=match decoder.decode(&packet) {
            Ok(value)=>value,
            Err(Error::DecodeError(_))=>continue,
            Err(e)=>return Err(format!("decoding audio: {e}")),
        };
        let spec=decoded.spec();
        let channels=spec.channels().count();
        if channels==0{return Err("decoded audio contains no channels".into());}
        if let Some(previous)=rate {
            if previous!=spec.rate(){return Err("sample rate changed midstream".into());}
        }
        rate=Some(spec.rate());
        let mut interleaved=vec![0.0f32;decoded.samples_interleaved()];
        decoded.copy_to_slice_interleaved(&mut interleaved);
        samples.extend(interleaved.chunks_exact(channels).map(|frame|
            frame.iter().sum::<f32>()/channels as f32));
    }
    let sample_rate=rate.ok_or("empty audio stream")?;
    if samples.is_empty(){return Err("no audio samples".into());}
    if samples.iter().any(|x|!x.is_finite()){return Err("non-finite decoded samples".into());}
    Ok(MonoAudio{samples,sample_rate})
}
