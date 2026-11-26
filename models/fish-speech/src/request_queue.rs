use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use fish_speech_core::cpu_weighted_ar::SelectionMode;
use fish_speech_core::error::FishError;
use fish_speech_core::prompt::ReferenceExample;
use fish_speech_core::sampling::SamplingParams;
use fish_speech_core::synthesis::{SynthesisEvent, SynthesisRequest, SynthesisSample};
use thiserror::Error;

use crate::model_runtime::{CpuFishModel, ModelPaths, ModelRuntimeError};

#[derive(Debug, Clone)]
pub struct OwnedSynthesisRequest {
    pub text: String,
    pub references: Vec<ReferenceExample>,
    pub num_samples: usize,
    pub num_codebooks: usize,
    pub iterative_prompt: bool,
    pub chunk_length: usize,
    pub max_context_length: usize,
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
}

impl OwnedSynthesisRequest {
    pub fn borrowed(&self) -> SynthesisRequest<'_> {
        SynthesisRequest {
            text: &self.text,
            references: &self.references,
            num_samples: self.num_samples,
            num_codebooks: self.num_codebooks,
            iterative_prompt: self.iterative_prompt,
            chunk_length: self.chunk_length,
            max_context_length: self.max_context_length,
            max_new_tokens: self.max_new_tokens,
            sampling: self.sampling,
        }
    }

    fn accounted_bytes(&self) -> Option<usize> {
        let mut bytes = self.text.len();
        for reference in &self.references {
            bytes = bytes.checked_add(reference.text.len())?;
            for codebook in &reference.acoustic_codes {
                bytes = bytes.checked_add(codebook.len().checked_mul(size_of::<u32>())?)?;
            }
        }
        Some(bytes)
    }
}

impl From<&SynthesisRequest<'_>> for OwnedSynthesisRequest {
    fn from(request: &SynthesisRequest<'_>) -> Self {
        Self {
            text: request.text.to_owned(),
            references: request.references.to_vec(),
            num_samples: request.num_samples,
            num_codebooks: request.num_codebooks,
            iterative_prompt: request.iterative_prompt,
            chunk_length: request.chunk_length,
            max_context_length: request.max_context_length,
            max_new_tokens: request.max_new_tokens,
            sampling: request.sampling,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct QueueLimits {
    pub max_pending: usize,
    pub max_pending_bytes: usize,
}

impl Default for QueueLimits {
    fn default() -> Self {
        Self {
            max_pending: 16,
            max_pending_bytes: 16 * 1024 * 1024,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum QueueError {
    #[error("queue admission is closed")]
    Closed,
    #[error("queue admission is paused during model replacement")]
    Paused,
    #[error("queue is full or request exceeds memory budget")]
    Full,
    #[error("another request is already executing")]
    Busy,
    #[error("request worker panicked; reload the model before accepting new jobs")]
    Faulted,
    #[error("invalid queue limits or request size")]
    Invalid,
    #[error("queue state mutex poisoned")]
    Poisoned,
}

pub type QueueResult<T> = std::result::Result<T, QueueError>;

struct Pending {
    id: u64,
    bytes: usize,
    request: OwnedSynthesisRequest,
    cancel: Arc<AtomicBool>,
}

struct Active {
    id: u64,
    cancel: Arc<AtomicBool>,
}

struct QueueState {
    limits: QueueLimits,
    next_id: u64,
    pending_bytes: usize,
    pending: VecDeque<Pending>,
    active: Option<Active>,
    paused: bool,
    closed: bool,
    faulted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueStatus {
    pub pending: usize,
    pub pending_bytes: usize,
    pub active: Option<u64>,
    pub paused: bool,
    pub closed: bool,
    pub faulted: bool,
}

#[derive(Clone)]
pub struct FishRequestQueue {
    state: Arc<Mutex<QueueState>>,
}

struct ActiveGuard {
    state: Arc<Mutex<QueueState>>,
    id: u64,
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            if state
                .active
                .as_ref()
                .is_some_and(|active| active.id == self.id)
            {
                state.active = None;
                if std::thread::panicking() {
                    state.faulted = true;
                    for pending in state.pending.drain(..) {
                        pending.cancel.store(true, Ordering::Release);
                    }
                    state.pending_bytes = 0;
                }
            }
        }
    }
}

pub struct QueuePauseGuard {
    state: Arc<Mutex<QueueState>>,
}

impl Drop for QueuePauseGuard {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.paused = false;
        }
    }
}

pub struct QueueOutcome<T> {
    pub id: u64,
    pub result: std::result::Result<T, ModelRuntimeError>,
}

impl FishRequestQueue {
    pub fn new(limits: QueueLimits) -> QueueResult<Self> {
        if limits.max_pending == 0 || limits.max_pending_bytes == 0 {
            return Err(QueueError::Invalid);
        }
        Ok(Self {
            state: Arc::new(Mutex::new(QueueState {
                limits,
                next_id: 1,
                pending_bytes: 0,
                pending: VecDeque::new(),
                active: None,
                paused: false,
                closed: false,
                faulted: false,
            })),
        })
    }

    pub fn enqueue(&self, request: OwnedSynthesisRequest) -> QueueResult<u64> {
        let bytes = request.accounted_bytes().ok_or(QueueError::Invalid)?;
        let mut state = self.state.lock().map_err(|_| QueueError::Poisoned)?;
        if state.closed {
            return Err(QueueError::Closed);
        }
        if state.faulted {
            return Err(QueueError::Faulted);
        }
        if state.paused {
            return Err(QueueError::Paused);
        }
        if state.pending.len() >= state.limits.max_pending
            || state
                .pending_bytes
                .checked_add(bytes)
                .is_none_or(|sum| sum > state.limits.max_pending_bytes)
        {
            return Err(QueueError::Full);
        }
        let id = state.next_id;
        state.next_id = state.next_id.checked_add(1).ok_or(QueueError::Invalid)?;
        state.pending_bytes += bytes;
        state.pending.push_back(Pending {
            id,
            bytes,
            request,
            cancel: Arc::new(AtomicBool::new(false)),
        });
        Ok(id)
    }

    pub fn cancel(&self, id: u64) -> QueueResult<bool> {
        let mut state = self.state.lock().map_err(|_| QueueError::Poisoned)?;
        if let Some(position) = state.pending.iter().position(|item| item.id == id) {
            let item = state
                .pending
                .remove(position)
                .expect("position came from queue");
            item.cancel.store(true, Ordering::Release);
            state.pending_bytes -= item.bytes;
            return Ok(true);
        }
        if let Some(active) = &state.active {
            if active.id == id {
                active.cancel.store(true, Ordering::Release);
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn status(&self) -> QueueResult<QueueStatus> {
        let state = self.state.lock().map_err(|_| QueueError::Poisoned)?;
        Ok(QueueStatus {
            pending: state.pending.len(),
            pending_bytes: state.pending_bytes,
            active: state.active.as_ref().map(|job| job.id),
            paused: state.paused,
            closed: state.closed,
            faulted: state.faulted,
        })
    }

    pub fn close(&self) -> QueueResult<()> {
        let mut state = self.state.lock().map_err(|_| QueueError::Poisoned)?;
        state.closed = true;
        for pending in state.pending.drain(..) {
            pending.cancel.store(true, Ordering::Release);
        }
        state.pending_bytes = 0;
        if let Some(active) = &state.active {
            active.cancel.store(true, Ordering::Release);
        }
        Ok(())
    }

    pub fn pause_for_replacement(&self) -> QueueResult<QueuePauseGuard> {
        let mut state = self.state.lock().map_err(|_| QueueError::Poisoned)?;
        if state.closed {
            return Err(QueueError::Closed);
        }
        if state.paused || state.active.is_some() || !state.pending.is_empty() {
            return Err(QueueError::Busy);
        }
        state.paused = true;
        Ok(QueuePauseGuard {
            state: self.state.clone(),
        })
    }

    pub fn reload_cpu(
        &self,
        model: &mut CpuFishModel,
        paths: &ModelPaths,
        seed: u64,
        mode: SelectionMode,
    ) -> std::result::Result<(), ReloadError> {
        let _pause = self.pause_for_replacement()?;
        model.reload(paths, seed, mode)?;
        self.state.lock().map_err(|_| QueueError::Poisoned)?.faulted = false;
        Ok(())
    }

    pub fn process_next_with<T, F>(&self, run: F) -> QueueResult<Option<QueueOutcome<T>>>
    where
        F: FnOnce(
            u64,
            &OwnedSynthesisRequest,
            &dyn Fn() -> bool,
        ) -> std::result::Result<T, ModelRuntimeError>,
    {
        let job = {
            let mut state = self.state.lock().map_err(|_| QueueError::Poisoned)?;
            if state.faulted {
                return Err(QueueError::Faulted);
            }
            if state.active.is_some() || state.paused {
                return Err(QueueError::Busy);
            }
            let Some(job) = state.pending.pop_front() else {
                return Ok(None);
            };
            state.pending_bytes -= job.bytes;
            state.active = Some(Active {
                id: job.id,
                cancel: job.cancel.clone(),
            });
            job
        };
        let _guard = ActiveGuard {
            state: self.state.clone(),
            id: job.id,
        };
        let cancelled = || job.cancel.load(Ordering::Acquire);
        let result = if cancelled() {
            Err(ModelRuntimeError::Synthesis(FishError::InvalidTransition(
                "synthesis cancelled",
            )))
        } else {
            run(job.id, &job.request, &cancelled)
        };
        Ok(Some(QueueOutcome { id: job.id, result }))
    }

    pub fn process_next_cpu<E>(
        &self,
        model: &mut CpuFishModel,
        streaming: bool,
        mut emit: E,
    ) -> QueueResult<Option<QueueOutcome<Vec<SynthesisSample>>>>
    where
        E: for<'a> FnMut(u64, SynthesisEvent<'a>) -> std::result::Result<(), FishError>,
    {
        self.process_next_with(|id, owned, cancelled| {
            let request = owned.borrowed();
            model.synthesize(&request, cancelled, streaming, |event| emit(id, event))
        })
    }
}

#[derive(Debug, Error)]
pub enum ReloadError {
    #[error("request queue: {0}")]
    Queue(#[from] QueueError),
    #[error("model reload: {0}")]
    Model(#[from] ModelRuntimeError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;

    fn request(text: &str) -> OwnedSynthesisRequest {
        OwnedSynthesisRequest {
            text: text.into(),
            references: vec![],
            num_samples: 1,
            num_codebooks: 8,
            iterative_prompt: false,
            chunk_length: 120,
            max_context_length: 4096,
            max_new_tokens: 64,
            sampling: SamplingParams::default(),
        }
    }

    #[test]
    fn fifo_queue_bounds_and_cancel_free_capacity() {
        let queue = FishRequestQueue::new(QueueLimits {
            max_pending: 2,
            max_pending_bytes: 8,
        })
        .unwrap();
        let first = queue.enqueue(request("abc")).unwrap();
        let second = queue.enqueue(request("def")).unwrap();
        assert_eq!(queue.enqueue(request("gh")), Err(QueueError::Full));
        assert_eq!(queue.cancel(first), Ok(true));
        assert_eq!(queue.status().unwrap().pending_bytes, 3);
        let third = queue.enqueue(request("gh")).unwrap();
        let mut seen = Vec::new();
        for _ in 0..2 {
            let item = queue
                .process_next_with(|id, req, _| {
                    seen.push((id, req.text.clone()));
                    Ok(())
                })
                .unwrap()
                .unwrap();
            item.result.unwrap();
        }
        assert_eq!(seen, vec![(second, "def".into()), (third, "gh".into())]);
        assert_eq!(queue.status().unwrap().pending_bytes, 0);
        assert!(queue.process_next_with(|_, _, _| Ok(())).unwrap().is_none());
    }

    #[test]
    fn cancel_active_from_another_thread_and_recover_queue() {
        let queue = FishRequestQueue::new(QueueLimits::default()).unwrap();
        let first = queue.enqueue(request("first")).unwrap();
        let second = queue.enqueue(request("second")).unwrap();
        let external = queue.clone();
        let (sender, receiver) = mpsc::channel();
        let thread = thread::spawn(move || {
            receiver.recv().unwrap();
            assert_eq!(external.cancel(first), Ok(true));
        });
        let first_result = queue
            .process_next_with(|id, _req, cancelled| {
                assert_eq!(id, first);
                sender.send(()).unwrap();
                while !cancelled() {
                    thread::yield_now();
                }
                Err::<(), _>(ModelRuntimeError::Synthesis(FishError::InvalidTransition(
                    "synthesis cancelled",
                )))
            })
            .unwrap()
            .unwrap();
        thread.join().unwrap();
        assert!(first_result.result.is_err());
        assert_eq!(queue.status().unwrap().active, None);
        assert_eq!(
            queue
                .process_next_with(|id, _, cancelled| {
                    assert_eq!(id, second);
                    assert!(!cancelled());
                    Ok(42)
                })
                .unwrap()
                .unwrap()
                .result
                .unwrap(),
            42
        );
    }

    #[test]
    fn only_one_active_and_reload_admission_is_atomic() {
        let queue = FishRequestQueue::new(QueueLimits::default()).unwrap();
        let id = queue.enqueue(request("one")).unwrap();
        let other = queue.clone();
        queue
            .process_next_with(|_, _, _| {
                assert_eq!(
                    other.process_next_with(|_, _, _| Ok(())).err(),
                    Some(QueueError::Busy)
                );
                assert_eq!(other.pause_for_replacement().err(), Some(QueueError::Busy));
                Ok(())
            })
            .unwrap()
            .unwrap()
            .result
            .unwrap();
        assert!(id > 0);
        let pause = queue.pause_for_replacement().unwrap();
        assert_eq!(other.enqueue(request("blocked")), Err(QueueError::Paused));
        drop(pause);
        assert!(other.enqueue(request("allowed")).is_ok());
    }

    #[test]
    fn close_marks_active_for_cancellation_and_drops_pending() {
        let queue = FishRequestQueue::new(QueueLimits::default()).unwrap();
        let _active = queue.enqueue(request("one")).unwrap();
        let other = queue.clone();
        queue
            .process_next_with(|_, _, cancelled| {
                other.enqueue(request("two")).unwrap();
                other.close().unwrap();
                assert!(cancelled());
                assert_eq!(other.status().unwrap().pending, 0);
                Err::<(), _>(ModelRuntimeError::Closed)
            })
            .unwrap()
            .unwrap();
        assert_eq!(queue.enqueue(request("three")), Err(QueueError::Closed));
    }

    #[test]
    fn panic_poisoning_cancels_pending_and_requires_model_reload() {
        let queue = FishRequestQueue::new(QueueLimits::default()).unwrap();
        queue.enqueue(request("abc")).unwrap();
        let other = queue.clone();
        assert!(
            std::panic::catch_unwind(move || {
                let _ = other.process_next_with(
                    |_, _, _| -> std::result::Result<(), ModelRuntimeError> { panic!("test only") },
                );
            })
            .is_err()
        );
        let status = queue.status().unwrap();
        assert_eq!(status.active, None);
        assert!(status.faulted);
        assert_eq!(status.pending, 0);
        assert_eq!(queue.enqueue(request("next")), Err(QueueError::Faulted));
        assert!(queue.pause_for_replacement().is_ok());
    }
}
