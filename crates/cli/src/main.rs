mod prepare;
mod replay;

use rl_mistake_analysis_candidates::{MistakeKind, ReplayFacts};
use rl_mistake_analysis_inference::{Model, manifest::Manifest};
use std::{
    env, fs,
    io::{self, Read},
    path::Path,
};

const HELP: &str = "Usage:
  rl-mistake-analysis events FILE.replay > events.json
  rl-mistake-analysis candidates < events.json
  rl-mistake-analysis contract KIND
  rl-mistake-analysis validate-annotations ANNOTATIONS.json
  rl-mistake-analysis prepare ANNOTATIONS.json REPLAY_DIRECTORY OUTPUT.parquet KIND
  rl-mistake-analysis validate-manifest DIRECTORY
  rl-mistake-analysis score MODEL_DIRECTORY KIND VERSION < events.json

events extracts subtr-actor events from a Rocket League replay.
candidates emits candidates for each implemented mistake kind.
contract emits a kind's feature schema and ordered feature names.
validate-annotations checks replay references and incident annotations.
prepare extracts features for annotated incidents and writes a Parquet table.
validate-manifest checks the model catalog and its declared artifact hashes.
score evaluates one kind with its supplied model.";

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let result = match args.as_slice() {
        [] => return Err(HELP.into()),
        [help] if help == "--help" || help == "-h" => {
            println!("{HELP}");
            return Ok(());
        }
        [command, path] if command == "events" => replay::events(&fs::read(path)?)?,
        [command] if command == "candidates" => {
            let replay = read_events()?;
            let batches = MistakeKind::ALL
                .into_iter()
                .map(|kind| kind.candidates(&replay))
                .collect::<Result<Vec<_>, _>>()?;
            serde_json::to_value(batches)?
        }
        [command, name] if command == "contract" => {
            let kind = MistakeKind::parse(name)?;
            serde_json::json!({
                "kind": kind.name(),
                "input_schema": kind.schema(),
                "feature_names": kind.feature_names(),
            })
        }
        [command, path] if command == "validate-annotations" => {
            let dataset = prepare::load_annotations(Path::new(path))?;
            serde_json::json!({"replays": dataset.replays.len(), "annotations": dataset.annotations.len()})
        }
        [command, annotations, recordings, output, name] if command == "prepare" => {
            let count = prepare::prepare(
                Path::new(annotations),
                Path::new(recordings),
                Path::new(output),
                MistakeKind::parse(name)?,
            )?;
            serde_json::json!({"kind": name, "examples": count})
        }
        [command, directory] if command == "validate-manifest" => {
            let directory = Path::new(directory);
            let manifest = Manifest::load(directory)?;
            manifest.verify_files(directory)?;
            serde_json::json!({"version": manifest.version, "models": manifest.models.len()})
        }
        [command, directory, name, version] if command == "score" => {
            let mut model = Model::load(Path::new(directory), name, version)?;
            let replay = read_events()?;
            let kind = MistakeKind::parse(name)?;
            serde_json::to_value(model.score(kind.candidates(&replay)?)?)?
        }
        _ => return Err(HELP.into()),
    };
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}

fn read_events() -> Result<ReplayFacts, Box<dyn std::error::Error>> {
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    Ok(serde_json::from_str(&text)?)
}

fn main() {
    if let Err(error) = run() {
        eprintln!("rl-mistake-analysis: {error}");
        std::process::exit(1);
    }
}
