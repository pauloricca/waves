# runtime (Rust POC)

A minimal runtime proof-of-concept for running Waves YAML sounds without the Python live tooling.

## Implemented node subset

- `osc` (`type: sin|sqr|saw`, numeric `freq`, numeric `amp`)
- `envelope` (`attack`, `decay`, `sustain`, `release`, `signal`)
- `mix` (arbitrary named child tracks)

## Current limitations

- Expression strings are not supported yet (e.g. `freq: "220 * i"`).
- Mono render only.
- No live reloading / visuals / MIDI.
- `--play` uses the default system output device and currently plays the rendered buffer for the requested duration; it is not a live-editing engine yet.

## Run locally

```bash
cd runtime
cargo run -- ../sounds/tests.yaml your_sound_name --seconds 4 --sample-rate 48000 --block-size 256 --write-wav out.wav
```

The runtime prints:
- wall-clock render time
- audio duration
- realtime load percentage (`wall/audio * 100`)
- per-node processing percentages by node type

These metrics are intended for quickly evaluating Raspberry Pi feasibility.

To play through the default audio device instead of only writing a file:

```bash
cd runtime
cargo run -- ./poc.yaml poc_pad --seconds 4 --play
```

## Build & run with Docker

### Build image

```bash
cd runtime
docker build -t waves-runtime:local .
```

### Run container directly

```bash
cd runtime
docker run --rm -v "$(pwd)/..:/work" waves-runtime:local \
  /work/runtime/poc.yaml poc_pad \
  --seconds 4 --sample-rate 48000 --block-size 256 \
  --write-wav /work/runtime/out.wav
```

### Run with Docker Compose

```bash
cd runtime
docker compose up --build
```

`docker-compose.yml` is preconfigured for the included POC patch at `/work/runtime/poc.yaml` with sound `poc_pad`. Update those args if you want to render a different patch.

### Extract a local executable with Docker Compose

```bash
cd runtime
docker compose run --rm extract-runtime
```

That writes a host executable to `runtime/runtime`, which you can run directly:

```bash
cd runtime
./runtime ./poc.yaml poc_pad --seconds 4 --sample-rate 48000 --block-size 256 --write-wav out.wav
```

### Extract a Raspberry Pi 4 binary with Docker Compose

For a Raspberry Pi 4 running a 64-bit Linux OS:

```bash
cd runtime
docker compose run --rm extract-runtime-pi4
```

That writes a Linux ARM64 executable to `runtime/runtime-linux-arm64`.

Copy that file to the Pi and run:

```bash
chmod +x ./runtime-linux-arm64
./runtime-linux-arm64 ./poc.yaml poc_pad --seconds 4 --sample-rate 48000 --block-size 256 --write-wav out.wav
```

Important:
- This target builds a `linux/arm64` binary.
- It is intended for Raspberry Pi 4 devices running a 64-bit OS.
- If your Pi is running a 32-bit OS, this binary will not run.

### Run the extracted executable

Usage:

```bash
./runtime <yaml-file> <sound-name> [--seconds N] [--sample-rate HZ] [--block-size N] [--write-wav PATH] [--play]
```

Arguments:
- `<yaml-file>`: path to the YAML file containing your sounds.
- `<sound-name>`: top-level sound key to render from that YAML file.

Options:
- `--seconds`: render duration in seconds. Default: `4`
- `--sample-rate`: output sample rate in Hz. Default: `48000`
- `--block-size`: processing block size. Default: `256`
- `--write-wav`: optional output WAV path. If omitted, the runtime only prints metrics and does not write audio.
- `--play`: play the rendered audio through the default output device on macOS or Linux.

Examples:

```bash
cd runtime
./runtime ./poc.yaml poc_pad --write-wav out.wav
```

```bash
cd runtime
./runtime ../sounds/tests.yaml your_sound_name --seconds 8 --sample-rate 44100 --block-size 512 --write-wav test.wav
```

```bash
cd runtime
./runtime ./poc.yaml poc_pad --seconds 4 --play
```

```bash
cd runtime
./runtime ./poc.yaml poc_pad --seconds 4 --play --write-wav out.wav
```

To print the CLI help:

```bash
cd runtime
./runtime --help
```

On Raspberry Pi and other Linux systems, `--play` uses the default ALSA output device. On macOS it uses the default CoreAudio output device.

## Delivery checklist

See [`CHECKLIST.md`](./CHECKLIST.md) for progress tracking toward the full runtime.
