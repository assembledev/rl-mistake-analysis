use std::{env, fs, path::Path};

fn main() {
    let directory = env::var("CARGO_MANIFEST_DIR").expect("missing Cargo manifest directory");
    let lock_path = Path::new(&directory).join("../../Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock_path.display());
    let source = fs::read_to_string(lock_path).expect("cannot read Cargo.lock");
    let lock: toml::Value = toml::from_str(&source).expect("invalid Cargo.lock");
    let supplier = lock["package"]
        .as_array()
        .expect("missing locked packages")
        .iter()
        .find(|package| package["name"].as_str() == Some("subtr-actor"))
        .expect("subtr-actor is not locked");
    let version = supplier["version"]
        .as_str()
        .expect("missing supplier version");
    let source = supplier["source"]
        .as_str()
        .expect("missing supplier source");
    println!("cargo:rustc-env=SUBTR_ACTOR_REVISION=subtr-actor {version} ({source})");
}
