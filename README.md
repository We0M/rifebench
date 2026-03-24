# rifebench

Focused benchmarking tool for **RIFE** interpolation models running on **VapourSynth R73+** with **vs-mlrt**.  
Automates test execution, NVIDIA GPU telemetry collection, and Markdown report generation.

- **Not universal** — targets only RIFE models for simplicity and comparability.
- **Windows + Linux** support.
- **Four components:** Runner → Parser → GPU Sampler → Reporter.

---

## Installation

Download latest release archive [Latest release](https://github.com/We0M/rifebench/releases/latest/)  
Unpack `.zip` to `python` dir.

```bash
pip install rifebench
```

**Required** (auto-installed):
- `pynvml` — NVIDIA GPU telemetry

**Optional:**
```bash
pip install ffmpeg-python   # source file media-info probing
```

### If use portable

1. **Download:**
    - [rifebench]()  
    - [pynvml](https://pypi.org/project/pynvml/#files)  
    - [ffmpeg-python](https://pypi.org/project/ffmpeg-python/#files)  
    - ffmpeg-python dependency [future](https://pypi.org/project/future/#files)  

2. **Open portable `python` dir:**
    - Unpack `rifebench_vXXX.zip`
    - Copy/move downloaded `pynvml_XXX.whl` and `ffmpeg-python_XXX.whl` to `./wheel`

3. **Install module:**  
```cmd
python pip.pyz install --no-index --find-links=./wheel/pynvml
python pip.pyz install --no-index --find-links=./wheel/ffmpeg-python
```

4. **Run(while in the Python directory):**  
```bash
# cmd
python -m rifebench.src --script ./bench.vpy --source D:/clip.mkv --models 425

# powershell
.\python -m rifebench.src --script ./bench.vpy --source D:/clip.mkv --models 425,4171
```

> [!TIP]
> For portable use `rifebench.src` instead of `rifebench`

---

## Quick Start

### Minimal (vspipe mode)
```bash
python -m rifebench --script ./bench.vpy --source D:/clip.mkv --models 47,425,417 
```

### FFmpeg pipe mode
```bash
python -m rifebench --script ./bench.vpy --source D:/clip.mkv --models 47,425 --mode ffmpeg
```

### Full example
```bash
python -m rifebench \
  --script ./my_bench.vpy \
  --source D:/clip.mkv \
  --models 47,425,417 \
  --mode vspipe \
  --multi 2 \
  --rife_v=2 \
  --streams 2 \
  --trt_fp16 1 \
  --format RGBH \
  --trt_out_fmt=1 \
  --use_cuda_graph 1 \
  --sc_threshold=0.10 \
  --gpu_id 0 \
  --warmup_frames 1000 \
  --warmup_runs 1 \
  --tests 3 \
  --shuffle 1 \
  --cooldown_s 4 \
  --output_dir ./bench_results \
  --bench_name my_bench \
  --period_ms 100
```

---

## CLI Reference

| Parameter          | Type   | Default | Description |
| ------------------ | ------ |---|----|
| `--source`         | path   | **required** | Path to source video file |
| `--models`         | string | **required** | Comma-separated RIFE model IDs, e.g. `47,425,417` |
| `--script`         | path   | **required** | Path to VapourSynth script |
| `--mode`           | string | `vspipe` | `vspipe` (render only) or `ffmpeg` (pipe through encoder) |
| `--format`         | string | `RGBH` | Pixel format: `RGBS` (fp32) or `RGBH` (fp16) |
| `--multi`          | int    | `2` | Frame multiplier passed to RIFE |
| `--rife_v`         | int    | `2` | RIFE version .onnx |
| `--streams`        | int    | `2` | Inference streams (vs-mlrt) |
| `--trt_fp16`       | 0┆1    | `1` | Enable TensorRT FP16 |
| `--trt_out_fmt`    | 0┆1    | `1` | output_format 0: fp32, 1: fp16 |
| `--use_cuda_graph` | 0┆1    | `1` | Enable CUDA Graph capture |
| `--gpu_id`         | int    | `0` | CUDA device index (exits if not found) |
| `--sc_threshold`   | FLOAT  | `0.10` | [Scene-cut threshold for RIFE](#sc_threshold) |
| `--warmup_frames`  | int    | `1000` | Frames for warmup (`0` = full clip) |
| `--warmup_runs`    | int    | `1` | Warmup iterations (results discarded) |
| `--tests`          | int    | `1` | Test iterations per model |
| `--shuffle`        | 0┆1    | `1` | Randomize model order between repetitions |
| `--cooldown_s`     | int    | `4` | Pause between runs (seconds) |
| `--output_dir`     | path   | `./bench_results` | Output directory |
| `--bench_name`     | string | `bench_YY-MM-DD_HH-MM` | Report file name (auto-suffixed `_01`, `_02`… on collision) |
| `--period_ms`      | int    | `100` | GPU sampling interval (ms) |
| `--command`        | string | *optional* | Custom ffmpeg command (part after `\|`). User is responsible for correctness |
| `--ffmpeg_path`    | path   | `ffmpeg` | Path to ffmpeg binary |
| `--vspipe_path`    | path   | `vspipe` | Path to vspipe binary |
| `--s_frame`        | int    | *optional* | First output frame (`vspipe -s N`) |
| `--e_frame`        | int    | *optional* | Last output frame (`vspipe -e N`) |

#### sc_threshold
Scene-cut threshold for RIFE  (lower=more sensitive, higher=less sensitive).  
Typical: 0.08-0.10 anime, 0.10-0.14 live-action, "0.14-0.20 noisy sources

---

## Modes

### 1. vspipe (default)

Renders frames via `vspipe` only, discarding output (`--`). All parameters are forwarded to the `.vpy` script:

```bash
vspipe -c y4m script.vpy -- \
  -a source=D:/clip.mkv \
  -a model=47 \
  -a gpu_id=0 \
  -a format=RGBH \
  -a multi=2 \
  -a rife_v=2 \
  -a streams=2 \
  -a trt_fp16=1 \
  -a trt_out_fmt=1 \
  -a use_cuda_graph=1 \
  -a sc_threshold=0.10
```

**Parsed output:** final stdout line → `Output N frames in N seconds (N fps)`

### 2. ffmpeg

Pipes `vspipe` y4m output into `ffmpeg`:

```bash
vspipe -c y4m script.vpy - -a ... | ffmpeg -benchmark -i - \
  -c:v hevc_nvenc -preset p5 -cq 18 \
  -profile:v main10 -pix_fmt p010le -f null NUL
```

- If `--command` is not provided, the default command above is used.
- If `--command` is provided, it replaces the part after `|`.
- The default uses `hevc_nvenc` / `main10` — may not match all clip formats.

**Parsed output:**
- Progress lines → `fps` collected into array for running median
- After `[out#` → `frames`, `fps`, `rtime` extracted from final summary

---

## .vpy Script Example

[**`Open a sample benchmark file`**](./bench.vpy)

rifebench passes parameters to your script via `vspipe -a key=value`. Your script must read them from `globals()`. Minimal example:

```python
...
# All values arrive as strings via `vspipe -a key=value`.

source         = globals().get('source', None)
model          = int(globals().get("model", "47"))
gpu_id         = int(globals().get("gpu_id", "0"))
fmt            = globals().get("format", "RGBH")       # "RGBS" | "RGBH"
multi          = int(globals().get("multi", "2"))
streams        = int(globals().get("streams", "2"))
trt_fp16       = bool(int(globals().get("trt_fp16", "1")))
use_cuda_graph = bool(int(globals().get("use_cuda_graph", "1")))
...
```

> **Key point:** `globals()` is a built-in dict populated by `vspipe -a` flags. All values are strings — cast them in your script.

---

## Architecture

```
┌──────────┐     ┌──────────┐     ┌────────────-─┐     ┌──────────┐
│  Runner  │────▶│  Parser  │────▶│ GPU Sampler  │────▶│ Reporter │
└──────────┘     └──────────┘     └────────────-─┘     └──────────┘
```

| Component | Module | Responsibility |
|---|---|---|
| **Runner**      | `runner.py` | Builds commands, launches `vspipe`/`ffmpeg` subprocesses, manages warmup/test iterations, shuffle, cooldown |
| **Parser**      | `parser.py` | Reads stdout/stderr line-by-line on the fly. Extracts `frames`, `fps`, `time_s`/`rtime`. Probes source & clip info |
| **GPU Sampler** | `sampler.py` | Background thread polling GPU via `pynvml` at `period_ms` intervals during test runs |
| **Reporter**    | `reporter.py` | Writes incremental `.md` reports with specs, summary (medians), per-run details. Auto-suffixes filenames |

---

## Output Format

Reports are saved as `{bench_name}.md` in `output_dir`.

### Specs Section

Collected automatically:

- **OS** — platform info (e.g. Windows 11 24H2)
- **GPU model, Driver, CUDA version** — via `nvidia-smi` / `pynvml`
- **TDP** (Power Limit), **Total/Used/Free VRAM** — via `pynvml`
- **CPU** — name, cores, clock
- **Bench Mode** — vspipe / ffmpeg
- **VapourSynth version** — parsed from `vspipe -v` (e.g. `Core R73`)
- **Source** — Width, Height, Frames, FPS, Format (via `ffmpeg-python` if available, else `N/A`)
- **Clip** — Width, Height, Frames, FPS, Format Name (via `vspipe -i`)

### Summary Table

Aggregated per-model results from all **successful** iterations:

| Model | FPS (avg┆median┆min) | Time s (avg┆mdn┆max) | GPU avg % | VRAM MB (mdn┆max) | PWR W (mdn┆max) | rxpci_max MB/s | txpci_max MB/s | Frames | Runs (tot┆succ) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

### All Runs Table

Per-run detail with GPU aggregates computed from samples within that run:

| Model | Run | Frames | FPS | Time s | GPU util % (avg┆p99) | MEM util % (avg┆p99) | VRAM Mb (avg┆max) | PWR W (avg┆max) | rxpci Mb/s (avg┆max) | txpci Mb/s (avg┆max) | gpu_clock | mem_clock |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |

- `gpu_util_avg` / `gpu_util_p99` — mean and 99th percentile of samples within the run
- `vram_max_mb` — maximum across samples within the run
- `gpu_clock` / `mem_clock` — median of samples within the run

### Errors Section

```markdown
## Errors
**Error model=47 run=2** | vspipe exited with code 1: "Filter error: ..."
```

---

## GPU Sampling

During each **test** run (not warmup), a background thread polls the GPU via `pynvml`:

| Metric       | Source                                   | Unit |
| ------------ | ---------------------------------------- | ---- |
| `gpu_util`   | `nvmlDeviceGetUtilizationRates`          | %    |
| `mem_util`   | `nvmlDeviceGetUtilizationRates`          | %    |
| `power_w`    | `nvmlDeviceGetPowerUsage / 1000`         | W    |
| `vram_mb`    | `nvmlDeviceGetMemoryInfo.used / 1048576` | MB   |
| `rxpci_mb_s` | `nvmlDeviceGetPcieThroughput (RX)`       | MB/s |
| `txpci_mb_s` | `nvmlDeviceGetPcieThroughput (TX)`       | MB/s |
| `gpu_clock`  | `nvmlDeviceGetClockInfo (SM)`            | MHz  |
| `mem_clock`  | `nvmlDeviceGetClockInfo (MEM)`           | MHz  |

Sampling interval is configurable via `--period_ms` (default: 100 ms).

---

## Error Handling

| Condition                                             | Result                                            |
| ----------------------------------------------------- | ------------------------------------------------- |
| Process exits with non-zero code                      | Iteration = **failed**                            |
| Cannot extract `frames` / `fps` / `time_s` or `rtime` | Iteration = **failed**                            |
| Cannot collect GPU metrics                            | Reported as `N/A`, iteration still **successful** |
| Failed iterations                                     | Excluded from Summary, listed in Errors section   |

---

## Project Structure

```
rifebench/
│   src/
│   ├── __init__.py        # Package init, version
│   ├── __main__.py        # Entry point (python -m rifebench)
│   ├── cli.py             # Argument parser
│   ├── runner.py          # Process execution & orchestration
│   ├── parser.py          # Output parsing & media probing
│   ├── sampler.py         # GPU telemetry (pynvml)
│   └── reporter.py        # Markdown report generation
├── pyproject.toml     # Package metadata
├── README.md          # This file
└── rifebench.html     # Offline documentation (EN/RU)
```

## MIT License