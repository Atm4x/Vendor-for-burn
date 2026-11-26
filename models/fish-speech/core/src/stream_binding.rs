use crate::graph_ir::ValueId;

/// Stable tensor binding identity for stream-oriented graph executors.
/// Temporary bindings denote tensors introduced by a backend adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Binding {
    Value(ValueId),
    Temp(usize),
}
