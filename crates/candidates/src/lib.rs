//! Rocket League mistake candidates and features from subtr-actor replay facts.

pub mod annotations;
pub mod input;
pub mod kind;
mod native_bump;

pub use input::ReplayFacts;
pub use kind::{Candidate, CandidateBatch, MistakeKind};
