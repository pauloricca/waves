use std::collections::HashMap;
use std::f32::consts::TAU;
use std::fs;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde_yaml::{Mapping, Value};

#[derive(Parser, Debug)]
#[command(name = "runtime")]
#[command(about = "Rust POC runtime for Waves YAML (subset: osc, envelope, mix)")]
struct Cli {
    /// Path to YAML file with sounds
    yaml: PathBuf,

    /// Top-level sound name in YAML
    sound: String,

    /// Render length in seconds
    #[arg(long, default_value_t = 4.0)]
    seconds: f32,

    /// Sample rate
    #[arg(long, default_value_t = 48_000)]
    sample_rate: u32,

    /// Processing block size
    #[arg(long, default_value_t = 256)]
    block_size: usize,

    /// Optional wav output path
    #[arg(long)]
    write_wav: Option<PathBuf>,

    /// Play the rendered audio through the default output device
    #[arg(long)]
    play: bool,
}

#[derive(Default)]
struct Profiler {
    total: Duration,
    by_node_type: HashMap<&'static str, Duration>,
}

impl Profiler {
    fn record(&mut self, node_type: &'static str, d: Duration) {
        self.total += d;
        *self.by_node_type.entry(node_type).or_default() += d;
    }
}

struct RenderCtx {
    sample_rate: f32,
    global_sample_index: usize,
}

enum Node {
    Osc(OscNode),
    Envelope(EnvelopeNode),
    Mix(MixNode),
}

impl Node {
    fn render_block(
        &mut self,
        frames: usize,
        ctx: &mut RenderCtx,
        profiler: &mut Profiler,
    ) -> Vec<f32> {
        let start = Instant::now();
        let (node_type, out) = match self {
            Node::Osc(node) => ("osc", node.render_block(frames, ctx)),
            Node::Envelope(node) => ("envelope", node.render_block(frames, ctx, profiler)),
            Node::Mix(node) => ("mix", node.render_block(frames, ctx, profiler)),
        };
        profiler.record(node_type, start.elapsed());
        out
    }
}

struct OscNode {
    freq: f32,
    amp: f32,
    phase: f32,
    wave_type: OscType,
}

#[derive(Clone, Copy)]
enum OscType {
    Sin,
    Sqr,
    Saw,
}

impl OscNode {
    fn render_block(&mut self, frames: usize, ctx: &RenderCtx) -> Vec<f32> {
        let dt = 1.0 / ctx.sample_rate;
        let mut out = Vec::with_capacity(frames);

        for _ in 0..frames {
            let sample = match self.wave_type {
                OscType::Sin => self.phase.sin(),
                OscType::Sqr => {
                    if self.phase.sin() >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    }
                }
                OscType::Saw => (self.phase / std::f32::consts::PI) - 1.0,
            };
            out.push(sample * self.amp);
            self.phase += TAU * self.freq * dt;
            if self.phase > TAU {
                self.phase -= TAU;
            }
        }

        out
    }
}

struct EnvelopeNode {
    signal: Box<Node>,
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    total_samples: usize,
}

impl EnvelopeNode {
    fn render_block(
        &mut self,
        frames: usize,
        ctx: &mut RenderCtx,
        profiler: &mut Profiler,
    ) -> Vec<f32> {
        let signal = self.signal.render_block(frames, ctx, profiler);
        let mut out = vec![0.0; frames];

        let a = (self.attack * ctx.sample_rate).max(0.0) as usize;
        let d = (self.decay * ctx.sample_rate).max(0.0) as usize;
        let r = (self.release * ctx.sample_rate).max(0.0) as usize;
        let sustain_start = a + d;
        let release_start = self.total_samples.saturating_sub(r);

        for i in 0..frames {
            let global = ctx.global_sample_index + i;
            let env = if a > 0 && global < a {
                global as f32 / a as f32
            } else if d > 0 && global < sustain_start {
                let t = (global - a) as f32 / d as f32;
                1.0 + (self.sustain - 1.0) * t
            } else if global < release_start {
                self.sustain
            } else if r > 0 && global < self.total_samples {
                let t = (global - release_start) as f32 / r as f32;
                self.sustain * (1.0 - t)
            } else {
                0.0
            };
            out[i] = signal[i] * env;
        }

        out
    }
}

struct MixNode {
    tracks: Vec<(String, Node)>,
}

impl MixNode {
    fn render_block(
        &mut self,
        frames: usize,
        ctx: &mut RenderCtx,
        profiler: &mut Profiler,
    ) -> Vec<f32> {
        let mut mixed = vec![0.0; frames];
        for (_, track) in &mut self.tracks {
            let signal = track.render_block(frames, ctx, profiler);
            for i in 0..frames {
                mixed[i] += signal[i];
            }
        }
        mixed
    }
}

fn parse_node(v: &Value, total_samples: usize) -> Result<Node> {
    let m = as_mapping(v)?;
    if m.len() != 1 {
        return Err(anyhow!("Each node must have exactly one key, got {m:?}"));
    }

    let (k, params) = m.iter().next().unwrap();
    let node_type = as_str(k)?;
    let params_map = as_mapping(params)?;

    match node_type {
        "osc" => parse_osc(params_map),
        "envelope" => parse_envelope(params_map, total_samples),
        "mix" => parse_mix(params_map, total_samples),
        other => Err(anyhow!(
            "Unsupported node '{other}' in POC runtime. Supported: osc, envelope, mix"
        )),
    }
}

fn parse_osc(params: &Mapping) -> Result<Node> {
    let wave_type = params
        .get(Value::String("type".to_string()))
        .map(|v| as_str(v).map(str::to_lowercase))
        .transpose()?
        .unwrap_or_else(|| "sin".to_string());

    let wave_type = match wave_type.as_str() {
        "sin" => OscType::Sin,
        "sqr" => OscType::Sqr,
        "saw" => OscType::Saw,
        other => return Err(anyhow!("Unsupported osc type '{other}' in POC")),
    };

    let freq = get_f32(params, "freq")?.unwrap_or(440.0);
    let amp = get_f32(params, "amp")?.unwrap_or(1.0);

    Ok(Node::Osc(OscNode {
        freq,
        amp,
        phase: 0.0,
        wave_type,
    }))
}

fn parse_envelope(params: &Mapping, total_samples: usize) -> Result<Node> {
    let signal_val = params
        .get(Value::String("signal".to_string()))
        .ok_or_else(|| anyhow!("envelope requires signal"))?;

    let signal = parse_node(signal_val, total_samples)?;

    Ok(Node::Envelope(EnvelopeNode {
        signal: Box::new(signal),
        attack: get_f32(params, "attack")?.unwrap_or(0.0),
        decay: get_f32(params, "decay")?.unwrap_or(0.0),
        sustain: get_f32(params, "sustain")?.unwrap_or(1.0),
        release: get_f32(params, "release")?.unwrap_or(0.0),
        total_samples,
    }))
}

fn parse_mix(params: &Mapping, total_samples: usize) -> Result<Node> {
    let mut tracks = Vec::new();

    for (k, v) in params {
        let track_name = as_str(k)?.to_string();
        if ["duration", "bpm", "context"].contains(&track_name.as_str()) {
            continue;
        }
        tracks.push((track_name, parse_node(v, total_samples)?));
    }

    if tracks.is_empty() {
        return Err(anyhow!("mix node has no tracks"));
    }

    Ok(Node::Mix(MixNode { tracks }))
}

fn as_mapping(v: &Value) -> Result<&Mapping> {
    v.as_mapping()
        .ok_or_else(|| anyhow!("Expected mapping, got {v:?}"))
}

fn as_str(v: &Value) -> Result<&str> {
    v.as_str()
        .ok_or_else(|| anyhow!("Expected string, got {v:?}"))
}

fn get_f32(map: &Mapping, key: &str) -> Result<Option<f32>> {
    let Some(v) = map.get(Value::String(key.to_string())) else {
        return Ok(None);
    };

    match v {
        Value::Number(n) => {
            let f = n.as_f64().ok_or_else(|| anyhow!("{key} must be numeric"))? as f32;
            Ok(Some(f))
        }
        Value::String(s) => Err(anyhow!(
            "{key} uses expression string '{s}', not yet supported in this POC"
        )),
        other => Err(anyhow!("{key} must be numeric, got {other:?}")),
    }
}

fn write_wav(path: &PathBuf, sample_rate: u32, samples: &[f32]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("Failed to create wav file at {}", path.display()))?;

    for s in samples {
        let scaled = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        writer.write_sample(scaled)?;
    }

    writer.finalize()?;
    Ok(())
}

fn render(cli: &Cli) -> Result<(Vec<f32>, Profiler, Duration, Duration)> {
    let yaml_text = fs::read_to_string(&cli.yaml)
        .with_context(|| format!("Failed to read {}", cli.yaml.display()))?;
    let root: Value = serde_yaml::from_str(&yaml_text)
        .with_context(|| format!("Failed to parse YAML from {}", cli.yaml.display()))?;

    let root_map = as_mapping(&root)?;
    let sound_val = root_map
        .get(Value::String(cli.sound.clone()))
        .ok_or_else(|| anyhow!("Sound '{}' not found in yaml", cli.sound))?;

    let total_samples = (cli.seconds * cli.sample_rate as f32) as usize;
    let mut root_node = parse_node(sound_val, total_samples)?;

    let mut profiler = Profiler::default();
    let mut ctx = RenderCtx {
        sample_rate: cli.sample_rate as f32,
        global_sample_index: 0,
    };

    let mut output = Vec::with_capacity(total_samples);
    let overall_start = Instant::now();

    while ctx.global_sample_index < total_samples {
        let remain = total_samples - ctx.global_sample_index;
        let frames = remain.min(cli.block_size);
        let block = root_node.render_block(frames, &mut ctx, &mut profiler);
        output.extend_from_slice(&block);
        ctx.global_sample_index += frames;
    }

    let wall = overall_start.elapsed();
    let audio_time = Duration::from_secs_f32(total_samples as f32 / cli.sample_rate as f32);

    Ok((output, profiler, wall, audio_time))
}

fn print_report(cli: &Cli, output: &[f32], profiler: &Profiler, wall: Duration, audio_time: Duration) {
    println!("Rendered '{}' from {}", cli.sound, cli.yaml.display());
    println!(
        "Samples: {} @ {} Hz ({} blocks of {})",
        output.len(),
        cli.sample_rate,
        output.len().div_ceil(cli.block_size),
        cli.block_size
    );
    println!("Wall time: {:.2?}", wall);
    println!("Audio time: {:.2?}", audio_time);

    let realtime_pct = (wall.as_secs_f64() / audio_time.as_secs_f64()) * 100.0;
    println!("Realtime load (wall/audio): {:.2}%", realtime_pct);

    if profiler.total.as_nanos() > 0 {
        println!("\nPer-node processing percentages:");
        let mut rows: Vec<_> = profiler.by_node_type.iter().collect();
        rows.sort_by_key(|(k, _)| *k);
        for (k, d) in rows {
            let pct = (d.as_secs_f64() / profiler.total.as_secs_f64()) * 100.0;
            println!("  {k:10} {:6.2}% ({:.2?})", pct, d);
        }
    }
}

fn play_audio(sample_rate: u32, samples: Vec<f32>) -> Result<()> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("No default output device available"))?;

    let supported = device
        .supported_output_configs()
        .context("Failed to query output configs")?
        .find(|cfg| {
            cfg.min_sample_rate().0 <= sample_rate
                && cfg.max_sample_rate().0 >= sample_rate
        })
        .ok_or_else(|| anyhow!("No output config supports sample rate {}", sample_rate))?;

    let channels = supported.channels();
    let config = supported.with_sample_rate(cpal::SampleRate(sample_rate)).config();
    let err_fn = |err| eprintln!("Audio stream error: {err}");
    let shared_samples = Arc::new(samples);
    let position = Arc::new(AtomicUsize::new(0));
    let total_frames = shared_samples.len();
    let sleep_step = Duration::from_millis(10);

    let stream = match supported.sample_format() {
        cpal::SampleFormat::F32 => build_stream::<f32>(
            &device,
            &config,
            channels,
            shared_samples.clone(),
            position.clone(),
            err_fn,
        )?,
        cpal::SampleFormat::I16 => build_stream::<i16>(
            &device,
            &config,
            channels,
            shared_samples.clone(),
            position.clone(),
            err_fn,
        )?,
        cpal::SampleFormat::U16 => build_stream::<u16>(
            &device,
            &config,
            channels,
            shared_samples.clone(),
            position.clone(),
            err_fn,
        )?,
        other => return Err(anyhow!("Unsupported sample format: {other:?}")),
    };

    stream.play().context("Failed to start output stream")?;
    while position.load(Ordering::Relaxed) < total_frames {
        thread::sleep(sleep_step);
    }
    thread::sleep(sleep_step);

    Ok(())
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    channels: u16,
    samples: Arc<Vec<f32>>,
    position: Arc<AtomicUsize>,
    err_fn: impl FnMut(cpal::StreamError) + Send + 'static,
) -> Result<cpal::Stream>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let channels = channels as usize;
    let stream = device.build_output_stream(
        config,
        move |data: &mut [T], _| {
            let mut frame_index = position.load(Ordering::Relaxed);
            for frame in data.chunks_mut(channels) {
                let sample = if frame_index < samples.len() {
                    samples[frame_index]
                } else {
                    0.0
                };
                let value = T::from_sample(sample);
                for out in frame {
                    *out = value;
                }
                if frame_index < samples.len() {
                    frame_index += 1;
                }
            }
            position.store(frame_index, Ordering::Relaxed);
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let (output, profiler, wall, audio_time) = render(&cli)?;
    print_report(&cli, &output, &profiler, wall, audio_time);

    if let Some(path) = &cli.write_wav {
        write_wav(path, cli.sample_rate, &output)?;
        println!("Wrote wav: {}", path.display());
    }

    if cli.play {
        println!("Playing audio on the default output device...");
        play_audio(cli.sample_rate, output)?;
    }

    Ok(())
}
