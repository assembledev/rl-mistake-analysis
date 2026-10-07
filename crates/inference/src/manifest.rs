//! Model catalog metadata and artifact integrity checks.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::Read,
    path::Path,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub schema_version: u32,
    pub version: String,
    pub models: Vec<ModelEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelEntry {
    pub kind: String,
    pub version: String,
    pub input_schema: String,
    pub feature_names: Vec<String>,
    pub runtime: String,
    pub model_file: String,
    pub evaluation_file: String,
    pub keep_threshold: f64,
    pub status: ReleaseStatus,
    pub provenance: Provenance,
    pub files: Vec<ArtifactFile>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReleaseStatus {
    Experimental,
    Approved,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Provenance {
    pub source_revision: String,
    pub engine_revisions: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFile {
    pub path: String,
    pub sha256: String,
}

impl Manifest {
    pub fn load(directory: &Path) -> Result<Self, String> {
        let path = directory.join("manifest.json");
        let bytes =
            fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        let manifest: Self = serde_json::from_slice(&bytes)
            .map_err(|error| format!("invalid {}: {error}", path.display()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 || self.version.trim().is_empty() || self.models.is_empty() {
            return Err("invalid manifest schema, version, or model list".into());
        }
        let mut identities = HashSet::new();
        let mut hashes = HashMap::new();
        for model in &self.models {
            model.validate()?;
            if !identities.insert((&model.kind, &model.version)) {
                return Err("duplicate model kind and version".into());
            }
            for file in &model.files {
                if let Some(hash) = hashes.insert(&file.path, &file.sha256)
                    && hash != &file.sha256
                {
                    return Err(format!("conflicting artifact hashes: {}", file.path));
                }
            }
        }
        Ok(())
    }

    pub fn model(&self, kind: &str, version: &str) -> Result<&ModelEntry, String> {
        self.models
            .iter()
            .find(|model| model.kind == kind && model.version == version)
            .ok_or_else(|| format!("model is not in this manifest: {kind} {version}"))
    }

    pub fn verify_files(&self, directory: &Path) -> Result<(), String> {
        self.validate()?;
        for model in &self.models {
            model.verify_files(directory)?;
        }
        Ok(())
    }
}

impl ModelEntry {
    pub fn validate(&self) -> Result<(), String> {
        let mut names = HashSet::new();
        if [&self.kind, &self.version, &self.input_schema, &self.runtime]
            .iter()
            .any(|value| value.trim().is_empty())
            || self.feature_names.is_empty()
            || self
                .feature_names
                .iter()
                .any(|name| name.trim().is_empty() || !names.insert(name))
            || !self.keep_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.keep_threshold)
            || self.provenance.source_revision.trim().is_empty()
            || self.provenance.engine_revisions.is_empty()
            || self
                .provenance
                .engine_revisions
                .iter()
                .any(|revision| revision.trim().is_empty())
        {
            return Err(format!("invalid model metadata for {}", self.kind));
        }
        let mut paths = HashSet::new();
        for file in &self.files {
            if file.path.contains('\\')
                || file
                    .path
                    .split('/')
                    .any(|part| matches!(part, "" | "." | ".."))
                || Path::new(&file.path).is_absolute()
                || !is_sha256(&file.sha256)
                || !paths.insert(&file.path)
            {
                return Err(format!("invalid or duplicate artifact path: {}", file.path));
            }
        }
        self.model_artifact()?;
        if self.model_file == self.evaluation_file
            || !self
                .files
                .iter()
                .any(|file| file.path == self.evaluation_file)
        {
            return Err("evaluation artifact must be declared separately from the model".into());
        }
        Ok(())
    }

    pub fn model_artifact(&self) -> Result<&ArtifactFile, String> {
        self.files
            .iter()
            .find(|file| file.path == self.model_file)
            .ok_or_else(|| format!("undeclared model artifact: {}", self.model_file))
    }

    pub fn verify_files(&self, directory: &Path) -> Result<(), String> {
        self.validate()?;
        for artifact in &self.files {
            artifact.verify(directory)?;
        }
        Ok(())
    }
}

impl ArtifactFile {
    pub fn read_verified(&self, directory: &Path) -> Result<Vec<u8>, String> {
        let path = directory.join(&self.path);
        let bytes =
            fs::read(&path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
        if format!("{:x}", Sha256::digest(&bytes)) != self.sha256 {
            return Err(format!("artifact hash mismatch: {}", path.display()));
        }
        Ok(bytes)
    }

    pub fn verify(&self, directory: &Path) -> Result<(), String> {
        let path = directory.join(&self.path);
        let mut file = File::open(&path)
            .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        let mut digest = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let count = file
                .read(&mut buffer)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        if format!("{:x}", digest.finalize()) != self.sha256 {
            return Err(format!("artifact hash mismatch: {}", path.display()));
        }
        Ok(())
    }
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
