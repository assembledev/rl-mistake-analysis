use crate::replay;
use parquet::{
    basic::Compression,
    data_type::{ByteArray, ByteArrayType, DataType, DoubleType, Int32Type, Int64Type},
    file::{
        metadata::KeyValue,
        properties::WriterProperties,
        writer::{SerializedFileWriter, SerializedRowGroupWriter},
    },
    schema::parser::parse_message_type,
};
use rl_mistake_analysis_candidates::{
    Candidate, MistakeKind, ReplayFacts,
    annotations::{Annotation, AnnotationDataset, IncidentAnchor},
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, path::Path, sync::Arc};

type Error = Box<dyn std::error::Error>;

pub fn load_annotations(path: &Path) -> Result<AnnotationDataset, Error> {
    let dataset: AnnotationDataset = serde_json::from_slice(&fs::read(path)?)?;
    dataset.validate()?;
    Ok(dataset)
}

struct Row<'a> {
    annotation: &'a Annotation,
    candidate: Candidate,
    engine_revision: String,
}

pub fn prepare(
    annotations: &Path,
    recordings: &Path,
    output: &Path,
    kind: MistakeKind,
) -> Result<usize, Error> {
    if output.extension().and_then(|value| value.to_str()) != Some("parquet") {
        return Err("prepared data must use a .parquet extension".into());
    }
    let mut dataset = load_annotations(annotations)?;
    let mut rows = Vec::new();
    for recording in &mut dataset.replays {
        let labels: Vec<_> = dataset
            .annotations
            .iter()
            .filter(|row| row.replay_sha256 == recording.sha256 && row.incident.kind() == kind)
            .collect();
        if labels.is_empty() {
            continue;
        }
        let path = recordings.join(format!("{}.replay", recording.sha256));
        let bytes =
            fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        if format!("{:x}", Sha256::digest(&bytes)) != recording.sha256 {
            return Err(format!("replay hash mismatch: {}", path.display()).into());
        }
        let facts: ReplayFacts = serde_json::from_value(replay::events(&bytes)?)?;
        if recording.rocket_league_id.is_some()
            && recording.rocket_league_id != facts.rocket_league_id
        {
            return Err(format!("intrinsic replay ID mismatch: {}", path.display()).into());
        }
        recording.rocket_league_id = facts.rocket_league_id.clone();
        let batch = kind.candidates(&facts)?;
        let mut incidents = HashMap::new();
        for candidate in batch.candidates {
            incidents.insert(candidate.anchor()?.key()?, candidate);
        }
        for annotation in labels {
            let key = annotation.incident.key()?;
            let candidate = incidents.remove(&key).ok_or_else(|| {
                format!("annotated incident is missing: {} {key}", recording.sha256)
            })?;
            if annotation.raw_time != candidate.raw_time {
                return Err(format!("annotated time disagrees with replay: {key}").into());
            }
            rows.push(Row {
                annotation,
                candidate,
                engine_revision: facts.revision.clone(),
            });
        }
    }
    if rows.is_empty() {
        return Err(format!("no annotations for {}", kind.name()).into());
    }
    write_table(output, kind, &dataset, &rows)?;
    Ok(rows.len())
}

fn write_column<T: DataType>(
    group: &mut SerializedRowGroupWriter<'_, &mut fs::File>,
    values: &[T::T],
    levels: Option<&[i16]>,
) -> Result<(), Error> {
    let mut column = group.next_column()?.ok_or("missing Parquet column")?;
    column.typed::<T>().write_batch(values, levels, None)?;
    column.close()?;
    Ok(())
}

fn write_table(
    output: &Path,
    kind: MistakeKind,
    dataset: &AnnotationDataset,
    rows: &[Row<'_>],
) -> Result<(), Error> {
    let feature_fields = kind
        .feature_names()
        .iter()
        .map(|name| format!("REQUIRED DOUBLE {name};"))
        .collect::<Vec<_>>()
        .join("\n");
    let schema = parse_message_type(&format!(
        "message examples {{
        REQUIRED BINARY replay_sha256 (UTF8);
        REQUIRED INT64 frame;
        REQUIRED BINARY initiator (UTF8);
        REQUIRED BINARY victim (UTF8);
        REQUIRED DOUBLE raw_time;
        REQUIRED INT32 label;
        OPTIONAL BINARY reject_reason (UTF8);
        REQUIRED BINARY engine_revision (UTF8);
        REQUIRED BINARY annotation_engine_revision (UTF8);
        REQUIRED BINARY source_event_ids (UTF8);
        {feature_fields}
    }}"
    ))?;
    let metadata = json!({
        "schema_version": 1,
        "kind": kind.name(),
        "input_schema": kind.schema(),
        "feature_names": kind.feature_names(),
        "replays": dataset.replays,
    });
    let properties = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(vec![KeyValue::new(
            "rl_mistake_analysis".into(),
            Some(metadata.to_string()),
        )]))
        .build();
    let output = std::path::absolute(output)?;
    let parent = output.parent().ok_or("output needs a file path")?;
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    {
        let mut writer = SerializedFileWriter::new(
            temporary.as_file_mut(),
            Arc::new(schema),
            Arc::new(properties),
        )?;
        for chunk in rows.chunks(8192) {
            let mut group = writer.next_row_group()?;
            let text: Vec<_> = chunk
                .iter()
                .map(|row| ByteArray::from(row.annotation.replay_sha256.as_str()))
                .collect();
            write_column::<ByteArrayType>(&mut group, &text, None)?;
            let frames = chunk
                .iter()
                .map(|row| i64::try_from(row.candidate.frame))
                .collect::<Result<Vec<_>, _>>()?;
            write_column::<Int64Type>(&mut group, &frames, None)?;
            for is_initiator in [true, false] {
                let participants: Vec<_> = chunk
                    .iter()
                    .map(|row| {
                        let IncidentAnchor::BumpingTeammate {
                            initiator, victim, ..
                        } = &row.annotation.incident;
                        ByteArray::from(
                            if is_initiator { initiator } else { victim }
                                .to_string()
                                .as_str(),
                        )
                    })
                    .collect();
                write_column::<ByteArrayType>(&mut group, &participants, None)?;
            }
            write_column::<DoubleType>(
                &mut group,
                &chunk
                    .iter()
                    .map(|row| row.candidate.raw_time)
                    .collect::<Vec<_>>(),
                None,
            )?;
            write_column::<Int32Type>(
                &mut group,
                &chunk
                    .iter()
                    .map(|row| i32::from(row.annotation.label))
                    .collect::<Vec<_>>(),
                None,
            )?;
            let reasons: Vec<_> = chunk
                .iter()
                .filter_map(|row| row.annotation.reject_reason.as_deref().map(ByteArray::from))
                .collect();
            let levels: Vec<_> = chunk
                .iter()
                .map(|row| i16::from(row.annotation.reject_reason.is_some()))
                .collect();
            write_column::<ByteArrayType>(&mut group, &reasons, Some(&levels))?;
            for values in [
                chunk
                    .iter()
                    .map(|row| row.engine_revision.clone())
                    .collect::<Vec<_>>(),
                chunk
                    .iter()
                    .map(|row| row.annotation.engine_revision.clone())
                    .collect(),
                chunk
                    .iter()
                    .map(|row| serde_json::to_string(&row.candidate.source_event_ids))
                    .collect::<Result<Vec<_>, _>>()?,
            ] {
                let text: Vec<_> = values
                    .iter()
                    .map(|value| ByteArray::from(value.as_str()))
                    .collect();
                write_column::<ByteArrayType>(&mut group, &text, None)?;
            }
            for index in 0..kind.feature_names().len() {
                let values: Vec<_> = chunk
                    .iter()
                    .map(|row| row.candidate.features[index])
                    .collect();
                write_column::<DoubleType>(&mut group, &values, None)?;
            }
            group.close()?;
        }
        writer.close()?;
    }
    temporary.as_file().sync_all()?;
    temporary.persist(&output)?;
    Ok(())
}
