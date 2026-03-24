import vapoursynth as vs
from vapoursynth import core
import vsmlrt
from vsmlrt import RIFE, RIFEModel, Backend
import os

# =============== Settings ===============

core.max_cache_size = 8192 #! 8Gb RAM

# ====== Get params from rifebench =======

fmt = globals().get("format", "RGBH") # vs.RGBS not good for realtime on GPU < RTX3080 if trt_fp16==False
fmt = vs.RGBH if fmt == "RGBH" else vs.RGBS
# RIFE
gpu_id         = int(globals().get("gpu_id", "0"))
model          = int(globals().get("model", "46"))
multi          = int(globals().get("multi", "2"))
# TensorRT
streams        = int(globals().get("streams", "2"))
trt_fp16       = bool(int(globals().get("trt_fp16", "1")))
use_cuda_graph = bool(int(globals().get("use_cuda_graph", "1")))
# added
rife_v         = int(globals().get("rife_v", "2")) # _implementation
trt_out_fmt    = int(globals().get("trt_out_fmt", "1")) # (0 - fp32,1 -fp16)
sc_threshold   = float(globals().get("sc_threshold", "0.1")) # Recommended range: 0.08-0.16

# RIFE.TRT requires sizes that are multiples of 32(sometimes 64 or 128).
if model in (4251, 4252):
    mod = 128
else:
    mod = 32

# ================ Source ================

src = globals().get('source', None)

if src and os.path.isfile(src):
    clip = core.lsmas.LWLibavSource(src)
else:
    raise ValueError( "Source not found!" )

# ================ Matrix ================

is_hd = clip.width > 1024 or clip.height >= 600
est_matrix = 1 if is_hd else 5

def ensure_matrix(clip):
    m = clip.get_frame(0).props.get('_Matrix')
    if m is None or m == 2:
        clip = core.std.SetFrameProps(clip, _Matrix=est_matrix)
    return clip

clip = ensure_matrix(clip)

# ========== Scene Change Detect =========

if clip.format.color_family in [vs.YUV, vs.GRAY]:
    sc_clip = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
else:
    sc_clip = core.resize.Bilinear(clip, format=vs.GRAY8, matrix=est_matrix)

sc_clip = sc_clip.misc.SCDetect(sc_threshold)

# ============== AddPadding ==============

pad_w = (-clip.width)  % mod
pad_h = (-clip.height) % mod

if pad_w or pad_h:
    clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)

# ================ Resize ================

clip = core.resize.Bicubic(clip, format=fmt)
clip = clip.std.CopyFrameProps(sc_clip, ["_SceneChangePrev", "_SceneChangeNext"])

# ============ Interpolation =============

clip = RIFE(
    clip,
    multi=multi,
    model=model,
    _implementation=rife_v,
    backend=Backend.TRT(
        fp16=trt_fp16,
        device_id=gpu_id,
        use_cuda_graph=use_cuda_graph,
        static_shape=True,
        num_streams=streams,
        output_format=trt_out_fmt,
    ),
)

# ============= Crop Padding =============

if pad_w or pad_h:
    clip = core.std.Crop(clip, right=pad_w, bottom=pad_h)

# ================ Output ================

clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix=est_matrix)
clip.set_output()