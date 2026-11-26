use crate::audio::SAMPLE_RATE;
use crate::conversation::{EncodedConversation, TokenEncoder};
use crate::error::{FishError, Result};
use crate::prompt::{LongPromptPlanner, ReferenceExample};
use crate::sampling::SamplingParams;
use crate::tokenizer::SpecialTokens;

pub trait SemanticGenerator {
    fn begin_sample(&mut self) -> Result<()>;
    fn generate(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        sampling: SamplingParams,
    ) -> Result<Vec<Vec<u32>>>;

    fn generate_with_cancel(
        &mut self,
        prompt: &EncodedConversation,
        max_new_tokens: usize,
        sampling: SamplingParams,
        is_cancelled: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<Vec<u32>>> {
        if is_cancelled() {
            return Err(FishError::InvalidTransition("synthesis cancelled"));
        }
        self.generate(prompt, max_new_tokens, sampling)
    }
}

pub trait AcousticDecoder {
    fn decode(&mut self, acoustic_codes: &[Vec<u32>]) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone)]
pub struct SynthesisRequest<'a> {
    pub text: &'a str,
    pub references: &'a [ReferenceExample],
    pub num_samples: usize,
    pub num_codebooks: usize,
    pub iterative_prompt: bool,
    pub chunk_length: usize,
    pub max_context_length: usize,
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
}

#[derive(Debug, Clone)]
pub struct DecodedSegment {
    pub text: String,
    pub samples: usize,
}

#[derive(Debug, Clone)]
pub struct SynthesisSample {
    pub sample_rate: usize,
    pub pcm: Vec<f32>,
    pub segments: Vec<DecodedSegment>,
}

#[derive(Debug)]
pub enum SynthesisEvent<'a> {
    Header {
        sample_rate: usize,
    },
    Segment {
        sample_index: usize,
        text: &'a str,
        pcm: &'a [f32],
    },
    Final {
        sample_index: usize,
        sample_rate: usize,
        pcm: &'a [f32],
    },
    Error {
        description: &'a str,
    },
}

pub fn synthesize<T, G, D, C>(
    tokenizer: &T,
    special: &SpecialTokens,
    request: &SynthesisRequest<'_>,
    generator: &mut G,
    decoder: &mut D,
    is_cancelled: C,
) -> Result<Vec<SynthesisSample>>
where
    T: TokenEncoder,
    G: SemanticGenerator,
    D: AcousticDecoder,
    C: FnMut() -> bool,
{
    synthesize_with_events(
        tokenizer,
        special,
        request,
        generator,
        decoder,
        is_cancelled,
        false,
        |_| Ok(()),
    )
}

pub fn synthesize_with_events<T, G, D, C, E>(
    tokenizer: &T,
    special: &SpecialTokens,
    request: &SynthesisRequest<'_>,
    generator: &mut G,
    decoder: &mut D,
    is_cancelled: C,
    streaming: bool,
    mut emit: E,
) -> Result<Vec<SynthesisSample>>
where
    T: TokenEncoder,
    G: SemanticGenerator,
    D: AcousticDecoder,
    C: FnMut() -> bool,
    E: for<'a> FnMut(SynthesisEvent<'a>) -> Result<()>,
{
    let result = synthesize_inner(
        tokenizer,
        special,
        request,
        generator,
        decoder,
        is_cancelled,
        streaming,
        &mut emit,
    );
    if let Err(err) = &result {
        let description = err.to_string();
        emit(SynthesisEvent::Error {
            description: &description,
        })?;
    }
    result
}

fn synthesize_inner<T, G, D, C, E>(
    tokenizer: &T,
    special: &SpecialTokens,
    request: &SynthesisRequest<'_>,
    generator: &mut G,
    decoder: &mut D,
    mut is_cancelled: C,
    streaming: bool,
    emit: &mut E,
) -> Result<Vec<SynthesisSample>>
where
    T: TokenEncoder,
    G: SemanticGenerator,
    D: AcousticDecoder,
    C: FnMut() -> bool,
    E: for<'a> FnMut(SynthesisEvent<'a>) -> Result<()>,
{
    request.sampling.validate()?;
    if request.num_samples == 0 {
        return Err(FishError::InvalidConfig("num_samples must be positive"));
    }
    let mut results = Vec::with_capacity(request.num_samples);
    if streaming {
        emit(SynthesisEvent::Header {
            sample_rate: SAMPLE_RATE,
        })?;
    }
    for sample_index in 0..request.num_samples {
        if is_cancelled() {
            return Err(FishError::InvalidTransition("synthesis cancelled"));
        }
        let mut planner = LongPromptPlanner::new(
            tokenizer,
            special,
            request.num_codebooks,
            request.text,
            request.iterative_prompt,
            request.chunk_length,
            request.max_context_length,
            request.references,
        )?;
        if planner.total_chunks() == 0 {
            return Err(FishError::InvalidShape("no text segments to synthesize"));
        }
        generator.begin_sample()?;
        let mut pcm = Vec::new();
        let mut segments = Vec::with_capacity(planner.total_chunks());
        while !planner.is_complete() {
            if is_cancelled() {
                return Err(FishError::InvalidTransition("synthesis cancelled"));
            }
            let next = planner
                .begin_next_chunk()?
                .ok_or(FishError::InvalidTransition("missing next text segment"))?;
            let generated = generator.generate_with_cancel(
                &next.encoded,
                request.max_new_tokens,
                request.sampling,
                &mut is_cancelled,
            )?;
            let codes = planner.accept_generated(generated)?;
            let decoded = decoder.decode(&codes.acoustic_codes)?;
            if decoded.iter().any(|s| !s.is_finite()) {
                return Err(FishError::InvalidShape("decoder returned non-finite PCM"));
            }
            if streaming {
                emit(SynthesisEvent::Segment {
                    sample_index,
                    text: &codes.text,
                    pcm: &decoded,
                })?;
            }
            segments.push(DecodedSegment {
                text: codes.text,
                samples: decoded.len(),
            });
            pcm.extend_from_slice(&decoded);
        }
        emit(SynthesisEvent::Final {
            sample_index,
            sample_rate: SAMPLE_RATE,
            pcm: &pcm,
        })?;
        results.push(SynthesisSample {
            sample_rate: SAMPLE_RATE,
            pcm,
            segments,
        });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct UtfEncoder;
    impl TokenEncoder for UtfEncoder {
        fn encode(&self, value: &str) -> Result<Vec<u32>> {
            Ok(value.chars().map(|c| c as u32).collect())
        }
    }

    #[derive(Default)]
    struct FakeGenerator {
        samples: usize,
        prompts: Vec<usize>,
    }
    impl SemanticGenerator for FakeGenerator {
        fn begin_sample(&mut self) -> Result<()> {
            self.samples += 1;
            Ok(())
        }
        fn generate(
            &mut self,
            prompt: &EncodedConversation,
            _max: usize,
            _params: SamplingParams,
        ) -> Result<Vec<Vec<u32>>> {
            self.prompts.push(prompt.length());
            Ok(vec![vec![10, 11, 12], vec![0, 1, 2], vec![0, 3, 4]])
        }
    }
    #[derive(Default)]
    struct FakeDecoder {
        calls: usize,
    }
    impl AcousticDecoder for FakeDecoder {
        fn decode(&mut self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
            self.calls += 1;
            assert_eq!(codes, &[vec![1, 2], vec![3, 4]]);
            Ok(vec![0.25, -0.5])
        }
    }

    fn request<'a>(text: &'a str) -> SynthesisRequest<'a> {
        SynthesisRequest {
            text,
            references: &[],
            num_samples: 1,
            num_codebooks: 2,
            iterative_prompt: true,
            chunk_length: 8,
            max_context_length: 2048,
            max_new_tokens: 4,
            sampling: SamplingParams::default(),
        }
    }

    struct CancelAwareGenerator;
    impl SemanticGenerator for CancelAwareGenerator {
        fn begin_sample(&mut self) -> Result<()> {
            Ok(())
        }
        fn generate(
            &mut self,
            _: &EncodedConversation,
            _: usize,
            _: SamplingParams,
        ) -> Result<Vec<Vec<u32>>> {
            panic!("synthesis must use the cancellable generator contract")
        }
        fn generate_with_cancel(
            &mut self,
            _: &EncodedConversation,
            _: usize,
            _: SamplingParams,
            is_cancelled: &mut dyn FnMut() -> bool,
        ) -> Result<Vec<Vec<u32>>> {
            assert!(is_cancelled());
            Err(FishError::InvalidTransition("synthesis cancelled"))
        }
    }

    #[test]
    fn cancellation_propagates_through_active_generator_not_only_chunk_boundary() {
        let mut polls = 0;
        let error = synthesize(
            &UtfEncoder,
            &SpecialTokens::new(100).unwrap(),
            &request("One sentence."),
            &mut CancelAwareGenerator,
            &mut FakeDecoder::default(),
            || {
                polls += 1;
                polls >= 3
            },
        )
        .unwrap_err();
        assert_eq!(error, FishError::InvalidTransition("synthesis cancelled"));
        assert_eq!(polls, 3);
    }

    #[test]
    fn joins_segments_into_one_final_audio() {
        let mut generator = FakeGenerator::default();
        let mut decoder = FakeDecoder::default();
        let result = synthesize(
            &UtfEncoder,
            &SpecialTokens::new(100).unwrap(),
            &request("Alpha. Beta. Gamma."),
            &mut generator,
            &mut decoder,
            || false,
        )
        .unwrap();
        let sample = &result[0];
        assert_eq!(sample.sample_rate, 44_100);
        assert!(sample.segments.len() >= 2);
        assert_eq!(generator.prompts.len(), decoder.calls);
        assert_eq!(sample.pcm.len(), decoder.calls * 2);
        assert!(sample.pcm.chunks_exact(2).all(|p| p == [0.25, -0.5]));
    }

    #[test]
    fn resets_generator_between_samples() {
        let mut generator = FakeGenerator::default();
        let mut decoder = FakeDecoder::default();
        let mut req = request("Test.");
        req.num_samples = 2;
        let result = synthesize(
            &UtfEncoder,
            &SpecialTokens::new(100).unwrap(),
            &req,
            &mut generator,
            &mut decoder,
            || false,
        )
        .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(generator.samples, 2);
        assert_eq!(result[0].pcm, result[1].pcm);
    }

    #[test]
    fn cancel_and_empty_input_fail_cleanly() {
        let mut generator = FakeGenerator::default();
        let mut decoder = FakeDecoder::default();
        let spec = SpecialTokens::new(100).unwrap();
        assert!(synthesize(
            &UtfEncoder,
            &spec,
            &request("  "),
            &mut generator,
            &mut decoder,
            || false
        )
        .is_err());
        assert!(synthesize(
            &UtfEncoder,
            &spec,
            &request("Test"),
            &mut generator,
            &mut decoder,
            || true
        )
        .is_err());
        assert_eq!(generator.samples, 0);
    }

    #[test]
    fn streaming_emits_header_segments_then_final() {
        let mut generator = FakeGenerator::default();
        let mut decoder = FakeDecoder::default();
        let mut sequence = Vec::<String>::new();
        let samples = synthesize_with_events(
            &UtfEncoder,
            &SpecialTokens::new(100).unwrap(),
            &request("Alpha. Beta. Gamma."),
            &mut generator,
            &mut decoder,
            || false,
            true,
            |event| {
                match event {
                    SynthesisEvent::Header { sample_rate } => {
                        assert_eq!(sample_rate, SAMPLE_RATE);
                        sequence.push("header".to_owned());
                    }
                    SynthesisEvent::Segment {
                        sample_index,
                        text,
                        pcm,
                    } => {
                        assert_eq!(sample_index, 0);
                        assert!(!text.is_empty());
                        assert_eq!(pcm, &[0.25, -0.5]);
                        sequence.push("segment".to_owned());
                    }
                    SynthesisEvent::Final {
                        sample_index,
                        sample_rate,
                        pcm,
                    } => {
                        assert_eq!(sample_index, 0);
                        assert_eq!(sample_rate, SAMPLE_RATE);
                        assert!(!pcm.is_empty());
                        sequence.push("final".to_owned());
                    }
                    SynthesisEvent::Error { description } => {
                        panic!("unexpected error: {description}")
                    }
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(sequence[0], "header");
        assert_eq!(sequence.last().unwrap(), "final");
        assert_eq!(
            sequence.iter().filter(|s| s.as_str() == "segment").count(),
            samples[0].segments.len()
        );
    }

    #[test]
    fn nonstreaming_emits_only_final_and_failures_emit_error() {
        let mut generator = FakeGenerator::default();
        let mut decoder = FakeDecoder::default();
        let mut events = Vec::new();
        synthesize_with_events(
            &UtfEncoder,
            &SpecialTokens::new(100).unwrap(),
            &request("Hello"),
            &mut generator,
            &mut decoder,
            || false,
            false,
            |event| {
                events.push(match event {
                    SynthesisEvent::Final { .. } => "final",
                    SynthesisEvent::Error { .. } => "error",
                    SynthesisEvent::Header { .. } => "header",
                    SynthesisEvent::Segment { .. } => "segment",
                });
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(events, vec!["final"]);
        let mut events = Vec::new();
        assert!(synthesize_with_events(
            &UtfEncoder,
            &SpecialTokens::new(100).unwrap(),
            &request("Hello"),
            &mut generator,
            &mut decoder,
            || true,
            true,
            |event| {
                events.push(match event {
                    SynthesisEvent::Header { .. } => "header",
                    SynthesisEvent::Error { .. } => "error",
                    _ => "unexpected",
                });
                Ok(())
            },
        )
        .is_err());
        assert_eq!(events, vec!["header", "error"]);
    }
}
