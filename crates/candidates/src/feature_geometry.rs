use crate::feature_input::{Body, Vector};
use subtr_actor::{
    BALL_COLLISION_RADIUS, STANDARD_GOAL_LINE_Y, STANDARD_GOAL_MOUTH_HALF_WIDTH_X,
    STANDARD_GOAL_MOUTH_HEIGHT_Z,
};

fn goal_y(team_zero: bool) -> f64 {
    f64::from(STANDARD_GOAL_LINE_Y) * if team_zero { 1.0 } else { -1.0 }
}

pub(crate) fn goal_opening_distance(
    position: Vector,
    attacking_team_zero: bool,
) -> Result<f64, String> {
    let distance = point_goal_opening_distance(position, attacking_team_zero)?;
    Ok((distance - f64::from(BALL_COLLISION_RADIUS)).max(0.0))
}

pub(crate) fn point_goal_opening_distance(
    position: Vector,
    attacking_team_zero: bool,
) -> Result<f64, String> {
    if ![position.x, position.y, position.z]
        .into_iter()
        .all(f64::is_finite)
    {
        return Err("nonfinite position".into());
    }
    let half_width = f64::from(STANDARD_GOAL_MOUTH_HALF_WIDTH_X);
    let height = f64::from(STANDARD_GOAL_MOUTH_HEIGHT_Z);
    let nearest = Vector {
        x: position.x.clamp(-half_width, half_width),
        y: goal_y(attacking_team_zero),
        z: position.z.clamp(0.0, height),
    };
    let distance = position.subtract(nearest).length();
    if !distance.is_finite() {
        return Err("nonfinite goal-opening distance".into());
    }
    Ok(distance)
}

pub(crate) fn directed_at_goal(
    body: &Body,
    attacking_team_zero: bool,
) -> Result<Option<bool>, String> {
    let Some(velocity) = body.linear_velocity else {
        return Ok(None);
    };
    if velocity.y == 0.0 {
        return Ok(Some(false));
    }
    let time = (goal_y(attacking_team_zero) - body.location.y) / velocity.y;
    if time < 0.0 {
        return Ok(Some(false));
    }
    let crossing = Vector {
        x: body.location.x + velocity.x * time,
        y: goal_y(attacking_team_zero),
        z: body.location.z + velocity.z * time,
    };
    Ok(Some(
        goal_opening_distance(crossing, attacking_team_zero)? == 0.0,
    ))
}
