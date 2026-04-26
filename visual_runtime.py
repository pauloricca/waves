from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import (
    VISUAL_RENDERER_FPS,
    VISUAL_RENDERER_HEIGHT,
    VISUAL_RENDERER_VSYNC,
    VISUAL_RENDERER_WIDTH,
)
from nodes.node_utils.base_node import BaseNode
from nodes.glsl import GLSLNode
from nodes.video import VideoNode


VERTEX_SHADER = """
#version 330
in vec2 in_position;
in vec2 in_uv;
out vec2 v_uv;

void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""


FRAGMENT_SHADER = """
#version 330
uniform sampler2D u_texture0;
uniform float u_time;
uniform float u_mix;
uniform float u_uniform_a;
uniform float u_uniform_b;
uniform float u_opacity;
uniform int u_shader_mode;
uniform vec2 u_resolution;

in vec2 v_uv;
out vec4 f_color;

vec2 rotate2d(vec2 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return mat2(c, -s, s, c) * p;
}

void main() {
    vec2 uv = v_uv;
    vec4 base = texture(u_texture0, uv);
    vec4 effected = base;

    if (u_shader_mode == 1) {
        effected = vec4(vec3(1.0) - base.rgb, 1.0);
    } else if (u_shader_mode == 2) {
        float pulse = 0.5 + 0.5 * sin((uv.x + uv.y + u_time) * max(0.5, u_uniform_b) * 6.28318);
        vec2 ripple_uv = uv + vec2(
            sin((uv.y + u_time) * max(0.5, u_uniform_b) * 4.0),
            cos((uv.x + u_time) * max(0.5, u_uniform_b) * 4.0)
        ) * (0.01 + u_uniform_a * 0.04);
        vec4 rippled = texture(u_texture0, clamp(ripple_uv, 0.0, 1.0));
        effected = vec4(mix(rippled.rgb, rippled.rgb * pulse, clamp(u_uniform_a, 0.0, 1.5)), 1.0);
    } else if (u_shader_mode == 3) {
        float luma = dot(base.rgb, vec3(0.2126, 0.7152, 0.0722));
        float edge = step(u_uniform_a, luma);
        effected = vec4(vec3(edge * max(u_uniform_b, 0.0)), 1.0);
    } else if (u_shader_mode == 4) {
        float amount = 0.003 + clamp(u_uniform_a, 0.0, 1.5) * 0.02;
        vec2 drift = vec2(cos(u_time * 0.7), sin(u_time * 0.9)) * amount;
        float r = texture(u_texture0, clamp(uv + drift, 0.0, 1.0)).r;
        float g = texture(u_texture0, uv).g;
        float b = texture(u_texture0, clamp(uv - drift, 0.0, 1.0)).b;
        effected = vec4(r, g, b, 1.0);
    } else if (u_shader_mode == 5) {
        float line_density = 180.0 + max(0.0, u_uniform_b) * 220.0;
        float line = sin(uv.y * line_density) * 0.5 + 0.5;
        float vignette = smoothstep(1.2, 0.15, distance(uv, vec2(0.5)));
        vec3 scanned = base.rgb * mix(0.65, 1.1, line) * mix(0.75, 1.0, vignette);
        effected = vec4(scanned, 1.0);
    } else if (u_shader_mode == 6) {
        vec2 centered = uv * 2.0 - 1.0;
        float segments = max(2.0, floor(3.0 + max(0.0, u_uniform_b) * 8.0));
        float angle = atan(centered.y, centered.x);
        float radius = length(centered);
        float sector = 6.28318 / segments;
        angle = mod(angle, sector);
        angle = abs(angle - sector * 0.5);
        vec2 mirrored = rotate2d(vec2(radius, 0.0), angle);
        vec2 sample_uv = clamp(mirrored * (0.85 + clamp(u_uniform_a, 0.0, 1.0) * 0.25) + 0.5, 0.0, 1.0);
        effected = texture(u_texture0, sample_uv);
    }

    vec3 color = mix(base.rgb, effected.rgb, clamp(u_mix, 0.0, 1.0));
    f_color = vec4(color, clamp(u_opacity, 0.0, 1.0));
}
"""


SHADER_MODE_MAP = {
    "passthrough": 0,
    "invert": 1,
    "pulse": 2,
    "threshold": 3,
    "rgb_split": 4,
    "scanlines": 5,
    "kaleido": 6,
}


def _set_uniform(program, name: str, value) -> None:
    try:
        program[name].value = value
    except KeyError:
        # GLSL compilers can optimize away uniforms that are declared but unused.
        return


def _extract_nodes(value: Any) -> list[BaseNode]:
    if isinstance(value, BaseNode):
        return [value]
    if isinstance(value, dict):
        nodes: list[BaseNode] = []
        for item in value.values():
            nodes.extend(_extract_nodes(item))
        return nodes
    if isinstance(value, (list, tuple, set)):
        nodes: list[BaseNode] = []
        for item in value:
            nodes.extend(_extract_nodes(item))
        return nodes
    return []


def _iter_child_nodes(node: BaseNode) -> list[BaseNode]:
    children: list[BaseNode] = []
    for value in vars(node).values():
        children.extend(_extract_nodes(value))
    return children


def _walk_nodes(node: BaseNode, visited: set[int] | None = None):
    if visited is None:
        visited = set()
    instance_id = id(node)
    if instance_id in visited:
        return
    visited.add(instance_id)
    yield node
    for child in _iter_child_nodes(node):
        yield from _walk_nodes(child, visited)


def has_visual_nodes(root_node: BaseNode) -> bool:
    return any(isinstance(node, (VideoNode, GLSLNode)) for node in _walk_nodes(root_node))


def collect_visual_snapshot(root_node: BaseNode) -> dict[str, Any] | None:
    source_info: dict[str, Any] | None = None
    shader_info: dict[str, Any] | None = None

    for node in _walk_nodes(root_node):
        if isinstance(node, VideoNode) and node.last_frame_info:
            source_info = dict(node.last_frame_info)
        elif isinstance(node, GLSLNode) and node.last_render_info:
            shader_info = dict(node.last_render_info)

    if source_info is None and shader_info is None:
        return None

    snapshot: dict[str, Any] = {}
    if source_info:
        snapshot.update(source_info)
    if shader_info:
        snapshot.update(shader_info)
    return snapshot


def _placeholder_frame(source_name: str | None, width: int, height: int, playhead: float, shader_name: str) -> np.ndarray:
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    source_seed = sum(ord(ch) for ch in (source_name or "missing")) % 255
    r = np.mod(xx * 255.0 + source_seed + playhead * 255.0, 255.0)
    g = np.mod(yy * 255.0 + source_seed * 0.5 + playhead * 96.0, 255.0)
    b = np.mod((1.0 - xx) * 255.0 + len(shader_name) * 24.0, 255.0)
    image = np.stack([r, g, b], axis=-1)

    stripe_period = max(8, width // 24)
    stripe_mask = ((np.arange(width) // stripe_period) % 2) == 0
    image[:, stripe_mask, :] *= 0.6

    return np.clip(image, 0.0, 255.0).astype(np.uint8)


@dataclass
class MediaInfo:
    duration: float | None
    frame_rate: float | None


class MediaStreamSession:
    def __init__(self, source_path: str, width: int, height: int, frame_rate: float):
        self.source_path = source_path
        self.width = width
        self.height = height
        self.frame_rate = max(frame_rate, 1.0)
        self.frame_size = width * height * 3
        self.process: subprocess.Popen | None = None
        self.current_frame_index = -1
        self.last_frame: np.ndarray | None = None
        self.recent_frames: dict[int, np.ndarray] = {}
        self.max_recent_frames = max(12, int(self.frame_rate * 3.0))

    def close(self) -> None:
        if self.process is None:
            return
        try:
            if self.process.stdout is not None:
                self.process.stdout.close()
        except Exception:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=0.5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass
        self.process = None

    def _start_process(self, start_time: float) -> None:
        self.close()
        command = [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            f"{max(start_time, 0.0):.3f}",
            "-i",
            self.source_path,
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"scale={self.width}:{self.height}:force_original_aspect_ratio=increase,crop={self.width}:{self.height}",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.current_frame_index = int(round(start_time * self.frame_rate)) - 1
        self.last_frame = None

    def _read_next_frame(self) -> np.ndarray | None:
        if self.process is None or self.process.stdout is None:
            return None
        try:
            raw = self.process.stdout.read(self.frame_size)
        except Exception:
            return None
        if len(raw) != self.frame_size:
            return None
        self.current_frame_index += 1
        self.last_frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        self.recent_frames[self.current_frame_index] = self.last_frame
        if len(self.recent_frames) > self.max_recent_frames:
            oldest_index = min(self.recent_frames.keys())
            self.recent_frames.pop(oldest_index, None)
        return self.last_frame

    def get_frame(self, target_time: float) -> np.ndarray | None:
        target_frame_index = int(round(max(target_time, 0.0) * self.frame_rate))

        cached_reverse_frame = self.recent_frames.get(target_frame_index)
        if cached_reverse_frame is not None:
            self.current_frame_index = target_frame_index
            self.last_frame = cached_reverse_frame
            return cached_reverse_frame

        if (
            self.process is None
            or target_frame_index < self.current_frame_index
            or target_frame_index - self.current_frame_index > int(self.frame_rate * 2.0)
        ):
            self._start_process(target_time)

        if self.last_frame is None:
            if self._read_next_frame() is None:
                return None

        while self.current_frame_index < target_frame_index:
            if self._read_next_frame() is None:
                break

        return self.last_frame


class MediaFrameProvider:
    def __init__(self):
        self._media_info_cache: dict[str, MediaInfo] = {}
        self._sessions: dict[tuple[str, int, int], MediaStreamSession] = {}

    def _probe(self, source_path: str) -> MediaInfo:
        if source_path in self._media_info_cache:
            return self._media_info_cache[source_path]

        duration = None
        frame_rate = None
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=avg_frame_rate:format=duration",
                    "-of",
                    "default=noprint_wrappers=1",
                    source_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.splitlines():
                key, _, value = line.partition("=")
                if not value:
                    continue
                if key == "duration":
                    duration = max(float(value), 0.0)
                elif key == "avg_frame_rate" and value != "0/0":
                    numerator, _, denominator = value.partition("/")
                    if denominator and float(denominator) != 0:
                        frame_rate = float(numerator) / float(denominator)
        except Exception:
            duration = None
            frame_rate = None

        media_info = MediaInfo(duration=duration, frame_rate=frame_rate)
        self._media_info_cache[source_path] = media_info
        return media_info

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()

    def load_frame(
        self,
        source_path: str | None,
        playhead: float,
        playhead_seconds: float | None,
        start_seconds: float,
        width: int,
        height: int,
        shader_name: str,
        loop: bool,
    ) -> np.ndarray:
        if not source_path or not os.path.exists(source_path):
            return _placeholder_frame(source_path, width, height, playhead, shader_name)

        media_info = self._probe(source_path)
        duration = media_info.duration if media_info.duration and media_info.duration > 0 else None
        if playhead_seconds is not None:
            target_time = max(float(start_seconds) + float(playhead_seconds), 0.0)
        else:
            target_time = max(float(start_seconds), 0.0) + (float(np.clip(playhead, 0.0, 1.0)) * duration if duration else 0.0)

        if duration:
            if loop:
                target_time = np.mod(target_time, duration)
            else:
                target_time = float(np.clip(target_time, 0.0, duration))

        frame_rate = media_info.frame_rate if media_info.frame_rate and media_info.frame_rate > 0 else 30.0
        session_key = (source_path, width, height)
        session = self._sessions.get(session_key)
        if session is None:
            session = MediaStreamSession(source_path, width, height, frame_rate)
            self._sessions[session_key] = session

        frame = session.get_frame(target_time)
        if frame is None:
            frame = _placeholder_frame(source_path, width, height, playhead, shader_name)
        return frame


class PlaybackController:
    def __init__(self):
        self.source_path: str | None = None
        self.clip_index: int | None = None
        self.playhead_seconds: float = 0.0
        self.last_update_time: float | None = None
        self.last_snapshot_time: float | None = None

    def resolve(self, snapshot: dict[str, Any], now: float) -> float | None:
        owner = snapshot.get("playhead_owner", "renderer")
        raw_clip_duration = snapshot.get("clip_duration")
        clip_duration = max(float(raw_clip_duration), 1e-6) if raw_clip_duration is not None else None
        if owner == "node":
            playhead_seconds = snapshot.get("playhead_seconds")
            if playhead_seconds is None:
                return None
            self.playhead_seconds = max(float(playhead_seconds), 0.0)
            if clip_duration is not None and bool(snapshot.get("loop", True)):
                self.playhead_seconds = float(np.mod(self.playhead_seconds, clip_duration))
            self.last_update_time = now
            self.last_snapshot_time = snapshot.get("snapshot_time")
            self.source_path = snapshot.get("source_path") or snapshot.get("source")
            self.clip_index = int(snapshot.get("clip_index", 0))
            return self.playhead_seconds

        source_path = snapshot.get("source_path") or snapshot.get("source")
        clip_index = int(snapshot.get("clip_index", 0))
        snapshot_time = snapshot.get("snapshot_time")

        if source_path != self.source_path or clip_index != self.clip_index:
            self.source_path = source_path
            self.clip_index = clip_index
            self.playhead_seconds = 0.0
            self.last_update_time = now
            self.last_snapshot_time = snapshot_time
            return self.playhead_seconds

        if snapshot_time is not None and snapshot_time != self.last_snapshot_time:
            self.last_snapshot_time = snapshot_time

        if self.last_update_time is None:
            self.last_update_time = now
            return self.playhead_seconds

        elapsed = max(0.0, now - self.last_update_time)
        self.last_update_time = now

        if bool(snapshot.get("paused", False)):
            return self.playhead_seconds

        speed = float(snapshot.get("speed", 1.0))
        self.playhead_seconds = max(self.playhead_seconds + elapsed * speed, 0.0)
        if clip_duration is not None and bool(snapshot.get("loop", True)):
            self.playhead_seconds = float(np.mod(self.playhead_seconds, clip_duration))
        elif clip_duration is not None:
            self.playhead_seconds = float(np.clip(self.playhead_seconds, 0.0, clip_duration))
        return self.playhead_seconds


def run_visual_renderer(
    snapshot_ref: list[dict[str, Any]],
    snapshot_lock: threading.Lock,
    should_stop_ref: list[bool],
    sound_name: str | None = None,
    video_fps_ref: list[float] | None = None,
):
    try:
        import glfw
        import moderngl
    except ModuleNotFoundError:
        print("Visual renderer disabled: install dependencies with `uv add moderngl glfw` and `uv sync`.")
        return

    if not glfw.init():
        print("Visual renderer disabled: failed to initialize GLFW.")
        return

    window = None
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        window = glfw.create_window(
            VISUAL_RENDERER_WIDTH,
            VISUAL_RENDERER_HEIGHT,
            f"Waves Visuals - {sound_name or 'Preview'}",
            None,
            None,
        )
        if window is None:
            print("Visual renderer disabled: failed to create GLFW window.")
            return

        glfw.make_context_current(window)
        glfw.swap_interval(1 if VISUAL_RENDERER_VSYNC else 0)

        ctx = moderngl.create_context()
        program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        vertices = np.array(
            [
                -1.0, -1.0, 0.0, 0.0,
                1.0, -1.0, 1.0, 0.0,
                -1.0, 1.0, 0.0, 1.0,
                1.0, 1.0, 1.0, 1.0,
            ],
            dtype="f4",
        )
        vbo = ctx.buffer(vertices.tobytes())
        vao = ctx.simple_vertex_array(program, vbo, "in_position", "in_uv")
        texture = ctx.texture((VISUAL_RENDERER_WIDTH, VISUAL_RENDERER_HEIGHT), 3)
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.use(0)

        frame_provider = MediaFrameProvider()
        playback_controller = PlaybackController()
        target_frame_time = 1.0 / max(1, VISUAL_RENDERER_FPS)
        start_time = time.time()

        while not glfw.window_should_close(window) and not should_stop_ref[0]:
            frame_start = time.time()
            glfw.poll_events()

            with snapshot_lock:
                snapshot = dict(snapshot_ref[0]) if snapshot_ref and snapshot_ref[0] else {}

            shader_name = str(snapshot.get("shader", "passthrough"))
            playhead = float(snapshot.get("playhead", 0.0))
            playhead_seconds = playback_controller.resolve(snapshot, frame_start)
            start_seconds = float(snapshot.get("start_seconds", 0.0))
            source_path = snapshot.get("source_path") or snapshot.get("source")
            loop = bool(snapshot.get("loop", True))

            frame = frame_provider.load_frame(
                source_path,
                playhead,
                playhead_seconds,
                start_seconds,
                VISUAL_RENDERER_WIDTH,
                VISUAL_RENDERER_HEIGHT,
                shader_name,
                loop,
            )
            texture.write(np.flipud(frame).tobytes())

            _set_uniform(program, "u_texture0", 0)
            _set_uniform(program, "u_time", time.time() - start_time)
            _set_uniform(program, "u_mix", float(snapshot.get("mix", 1.0)))
            _set_uniform(program, "u_uniform_a", float(snapshot.get("uniform_a", 0.5)))
            _set_uniform(program, "u_uniform_b", float(snapshot.get("uniform_b", 1.0)))
            _set_uniform(program, "u_opacity", float(snapshot.get("opacity", 1.0)))
            _set_uniform(program, "u_shader_mode", SHADER_MODE_MAP.get(shader_name, 0))
            _set_uniform(program, "u_resolution", (VISUAL_RENDERER_WIDTH, VISUAL_RENDERER_HEIGHT))

            ctx.clear(0.02, 0.02, 0.03, 1.0)
            vao.render(moderngl.TRIANGLE_STRIP)
            glfw.swap_buffers(window)

            elapsed = time.time() - frame_start
            if video_fps_ref is not None:
                video_fps_ref[0] = 1.0 / elapsed if elapsed > 0 else 0.0
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
    except Exception as exc:
        print(f"Visual renderer error: {exc}")
    finally:
        should_stop_ref[0] = True
        if video_fps_ref is not None:
            video_fps_ref[0] = 0.0
        try:
            frame_provider.close()
        except Exception:
            pass
        if window is not None:
            try:
                import glfw

                glfw.destroy_window(window)
            except Exception:
                pass
        try:
            import glfw

            glfw.terminate()
        except Exception:
            pass
