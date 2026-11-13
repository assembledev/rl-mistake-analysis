use crate::feature_input::{Metadata, PlayerData};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct ParticipantSamples {
    pub before_frame: Option<usize>,
    pub before_time: Option<f64>,
    pub peak_frame: Option<usize>,
    pub peak_time: Option<f64>,
    pub after_frame: Option<usize>,
    pub after_time: Option<f64>,
    pub status: SamplingStatus,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStatus {
    Complete,
    MissingLookbehind,
    MissingPlayerFrame,
    MissingVelocity,
    NonpositiveTimeStep,
    PlayBoundary,
    ReplayEnd,
}

impl ParticipantSamples {
    fn finish(
        &mut self,
        frames: &[Metadata],
        peak: Option<usize>,
        after: Option<usize>,
        status: SamplingStatus,
    ) {
        self.peak_frame = peak;
        self.peak_time = peak.map(|frame| frames[frame].time);
        self.after_frame = after;
        self.after_time = after.map(|frame| frames[frame].time);
        self.status = status;
    }
}

fn acceleration(
    player: &PlayerData,
    frames: &[Metadata],
    frame: usize,
    play_start: f64,
) -> Result<f64, SamplingStatus> {
    let previous = frame
        .checked_sub(1)
        .ok_or(SamplingStatus::MissingLookbehind)?;
    if frames[previous].time < play_start {
        return Err(SamplingStatus::PlayBoundary);
    }
    let dt = frames[frame].time - frames[previous].time;
    if dt <= 0.0 {
        return Err(SamplingStatus::NonpositiveTimeStep);
    }
    let velocity = |frame: usize| {
        player.frames[frame]
            .body()
            .ok_or(SamplingStatus::MissingPlayerFrame)?
            .linear_velocity
            .ok_or(SamplingStatus::MissingVelocity)
    };
    let delta = velocity(frame)?.subtract(velocity(previous)?);
    Ok(delta.x.hypot(delta.y).hypot(delta.z) / dt)
}

pub(crate) fn measure(
    player: &PlayerData,
    frames: &[Metadata],
    bump_frame: usize,
    play_start: f64,
    play_end: f64,
) -> ParticipantSamples {
    let before_frame = bump_frame
        .checked_sub(1)
        .filter(|&frame| frames[frame].time >= play_start);
    let mut samples = ParticipantSamples {
        before_frame,
        before_time: before_frame.map(|frame| frames[frame].time),
        peak_frame: None,
        peak_time: None,
        after_frame: None,
        after_time: None,
        status: SamplingStatus::MissingLookbehind,
    };
    if before_frame.is_none() {
        return samples;
    }
    let mut peak = None;
    for frame in bump_frame..frames.len() - 1 {
        if frames[frame + 1].time > play_end {
            samples.finish(frames, peak, None, SamplingStatus::PlayBoundary);
            return samples;
        }
        let measurements = [frame - 1, frame, frame + 1]
            .map(|frame| acceleration(player, frames, frame, play_start));
        let [before, current, after] = match measurements {
            [Ok(before), Ok(current), Ok(after)] => [before, current, after],
            results => {
                let status = results.into_iter().find_map(Result::err).unwrap();
                samples.finish(frames, peak, None, status);
                return samples;
            }
        };
        if peak.is_none() && current >= before && current > after {
            peak = Some(frame);
        } else if peak.is_some() && current <= before && current < after {
            samples.finish(frames, peak, Some(frame), SamplingStatus::Complete);
            return samples;
        }
    }
    samples.finish(frames, peak, None, SamplingStatus::ReplayEnd);
    samples
}
