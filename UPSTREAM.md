# Upstream dependencies

[subtr-actor](https://github.com/rlrml/subtr-actor) extracts gameplay events and replay context. [boxcars](https://github.com/nickbabcock/boxcars) reads the binary Rocket League recording. The offline preparation tool uses both; the reusable candidate and inference libraries accept extracted event records.

`Cargo.toml` declares these dependencies. `Cargo.lock` records their resolved versions and sources. Event exports, annotations, and model metadata preserve the supplier revision as provenance. Model compatibility depends on the feature contract, not an exact supplier revision.
