//! Rocket League mistake candidates and features from subtr-actor replay facts.

pub mod annotations;
pub mod bang_features;
mod bump_contacts;
pub mod bump_features;
mod feature_context;
pub mod feature_contract;
mod feature_geometry;
mod feature_input;
mod feature_measurements;
mod feature_values;
mod impact_sampling;
pub mod input;
pub mod kind;
mod native_bump;
mod native_touch;
mod overcommitting;
pub mod overcommitting_features;

pub use input::ReplayFacts;
pub use kind::{Candidate, CandidateBatch, KindDefinition, MistakeKind};
