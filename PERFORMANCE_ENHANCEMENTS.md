# Performance Enhancement Opportunities

This document summarizes concrete optimisation ideas discovered while reviewing the current code base. Each suggestion links back to the exact implementation it targets so the changes can be prioritised and implemented later on.

## Real-time rendering loop (`waves.py`)
* [x] **Avoid per-chunk allocations inside the audio callback.** `np.clip` and `np.concatenate` create fresh arrays on every audio block, and zero-padding allocates new buffers each time the callback underruns. Replacing them with in-place clipping via `np.clip(audio_data, -1.0, 1.0, out=audio_data)` and reusing a pre-sized scratch buffer for padding would reduce GC pressure in the callback’s hot path.【F:waves.py†L394-L410】
* [x] **Reduce deque append overhead for recordings.** `recording_buffer.extend(audio_data)` stores every sample (or frame) individually, which is costly when the callback runs at the audio rate. Appending contiguous NumPy blocks (e.g., `recording_buffer.append(audio_data.copy())`) and concatenating once during export would drastically shrink deque churn and Python-level bookkeeping.【F:waves.py†L381-L390】【F:waves.py†L209-L233】
* [x] **Cache stereo detection and reshape work.** The callback recomputes whether the signal is stereo for every chunk via `check_is_stereo` and repeatedly reshapes mono buffers. If the render context already knows the channel count, keeping that flag on the node and only re-evaluating when hot reload swaps in a new node would remove per-buffer Python branching.【F:waves.py†L362-L373】【F:waves.py†L284-L325】

## Trigger detection (`utils.py`)
* [x] **Vectorise rising-edge search.** `detect_triggers` iterates sample-by-sample in Python, even though NumPy can compute threshold crossings with boolean masks and `np.nonzero`, yielding a 10–20× speedup on large buffers. Maintaining the “last value” guard is straightforward by prepending that scalar before the vectorised comparison.【F:utils.py†L12-L45】

## Waveform visualiser (`utils.py`)
* [x] **Replace nested Python loops with NumPy reductions.** The current visualiser walks every column and row in Python to build ASCII art, which becomes expensive for wide terminals. Collapsing the grouping step with `wave.reshape` (after trimming) and computing the min/max columns in NumPy drops complexity and allows pre-formatting the string buffer only once per refresh.【F:utils.py†L102-L178】
* [x] **Cache terminal size queries.** `shutil.get_terminal_size()` is called every refresh; storing the last seen width/height and only refreshing on change would remove an expensive system call from the tight loop.【F:utils.py†L110-L176】

## Sound library loading (`sound_library.py`)
* [x] **Avoid redundant YAML parsing when hot reloading.** `load_all_sound_libraries` reads every YAML file twice (raw load then validation) and rebuilds `sound_index` from scratch on each invocation. Caching parsed ASTs per file and only re-validating the file that changed during hot reload would shorten reload times, especially with many libraries.【F:sound_library.py†L281-L356】
* [x] **Reuse a shared YAML loader.** The function re-resolves the `CSafeLoader` vs `SafeLoader` choice every time. Initialising the loader once at module import and reusing it would save repeated attribute checks during reload storms.【F:sound_library.py†L296-L333】

## Track mixing (`nodes/tracks.py`)
* [x] **Preallocate mix buffers and reuse padding.** `TracksNode.get_track_outputs` and `_do_render` repeatedly create temporary arrays (`np.zeros`, `np.vstack`) per track to pad signals. Switching to pre-sized arrays (e.g., allocating the `mixed` buffer once per chunk and filling slices) and using `np.copyto` for shorter tracks would minimise allocations during dense arrangements.【F:nodes/tracks.py†L125-L220】
* [x] **Share panning envelopes across tracks.** Dynamic panning nodes are rendered independently for every track even when they share identical parameters. Introducing a small cache keyed by node ID or parameter tuple would let multiple tracks reuse the same rendered automation curve within a block.【F:nodes/tracks.py†L145-L168】

Implementing these items should lower per-buffer latency, shorten hot-reload pauses, and improve the responsiveness of visual tooling without altering audible output.
