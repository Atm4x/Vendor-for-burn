use super::{candidates::infer_edits, features::EVIDENCE_FEATURES, model::EvidenceNet,
    repair::{RepairCandidate, RepairConfig, RepairInfo, apply_repair, input_notes}};
use crate::transcriber::{DecodedNote, Transcriber};

impl EvidenceNet {
    pub fn repair_notes(&self, notes: &[DecodedNote], evidence: &[[f32; EVIDENCE_FEATURES]])
        -> Result<(Vec<RepairCandidate>, RepairInfo), String>
    {
        if notes.len() != evidence.len() { return Err("evidence and notes must align".into()); }
        let sorted = input_notes(notes)?;
        let ordered = sorted.iter().map(|n| evidence[n.original_index]).collect::<Vec<_>>();
        let (edits, insertions) = infer_edits(&sorted, &ordered, |window, ev, start| {
            self.infer(window, ev, start)
        })?;
        apply_repair(&sorted, &edits, &insertions, RepairConfig {
            keep_threshold: self.config().keep_threshold,
            insert_threshold: self.config().insert_threshold,
            ..RepairConfig::default()
        })
    }

    pub fn transcribe_repaired(&self, transcriber: &Transcriber, audio16k: &[f32])
        -> Result<(Vec<RepairCandidate>, RepairInfo), String>
    {
        let (notes, ev) = transcriber.notes_with_evidence(audio16k)?;
        self.repair_notes(&notes, &ev)
    }
}
