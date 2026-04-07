# Runtime roadmap checklist

## Foundation

- [x] Create standalone runtime project (Rust).
- [x] Add CLI for YAML path, sound name, render length, sample rate, block size.
- [x] Add block-based render loop.
- [x] Add optional WAV output.
- [x] Add Dockerfile for reproducible builds.
- [x] Add docker-compose setup for quick local runs.

## YAML + node engine

- [x] Parse top-level sound from YAML.
- [x] Implement `osc` node.
- [x] Implement `envelope` node.
- [x] Implement `mix` node.
- [x] Fail fast on unsupported node types.
- [ ] Support node references/sub-patches across files.
- [ ] Support reusable IDs/reference behavior.
- [ ] Match Python YAML restructuring edge-cases.

## Expressions (core gap)

- [ ] Add expression parser/compiler for scalar+buffer operations.
- [ ] Support scalar-scalar, scalar-buffer, buffer-buffer math.
- [ ] Add function library parity (`sin`, `cos`, `clip`, `pow`, etc.).
- [ ] Add context vars (`time`, `n`, `i`, `item`, user vars, note constants).
- [ ] Add expression safety limits and deterministic behavior.

## Audio/runtime parity

- [x] Print performance metrics (wall/audio + per-node percentages).
- [x] Add realtime audio output backend (macOS + Raspberry Pi Linux).
- [ ] Add stereo pipeline support.
- [ ] Add latency/buffer tuning flags.
- [ ] Add clipping/normalization strategy parity.

## Packaging + deployment

- [ ] Cross-compile release binaries for Raspberry Pi targets.
- [ ] Publish minimal runtime image for Pi deployment.
- [ ] Add CI build/test pipeline.

## Validation

- [ ] Golden tests comparing Python vs runtime outputs.
- [ ] Benchmark suite for desktop and Raspberry Pi.
- [ ] Stress tests for long-running realtime sessions.
