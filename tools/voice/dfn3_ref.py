"""Référence f32 DeepFilterNet3 : classes officielles (extraites par AST du dépôt
Rikorose/DeepFilterNet) + features libDF réécrites en numpy depuis lib.rs.

Usage : dfn3_ref.py <src_dfn> <ckpt> <in.wav 48k> <out_dir>
Ecrit : out_dir/golden.safetensors (entrée, features, sorties) et weights.safetensors.
"""
import ast
import configparser
import math
import sys
import types
from functools import partial
from typing import Callable, Final, Iterable, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn import init
from torch.nn.parameter import Parameter

SRC, CKPT, WAV, OUT = sys.argv[1:5]
CFG = configparser.ConfigParser()
CFG.read(CKPT.rsplit("/checkpoints/", 1)[0] + "/config.ini")


class ModelParams:
    """Remplace df.config : mêmes champs que DfParams + deepfilternet3.ModelParams."""

    def __init__(self):
        d, n = CFG["df"], CFG["deepfilternet"]
        self.sr = d.getint("sr")
        self.fft_size = d.getint("fft_size")
        self.hop_size = d.getint("hop_size")
        self.nb_erb = d.getint("nb_erb")
        self.nb_df = d.getint("nb_df")
        self.norm_tau = d.getfloat("norm_tau")
        self.lsnr_max = d.getint("lsnr_max")
        self.lsnr_min = d.getint("lsnr_min")
        self.min_nb_freqs = d.getint("min_nb_erb_freqs")
        self.df_order = d.getint("df_order")
        self.df_lookahead = d.getint("df_lookahead")
        csv = lambda s: [int(x) for x in s.split(",")]
        self.conv_lookahead = n.getint("conv_lookahead")
        self.conv_ch = n.getint("conv_ch")
        self.conv_depthwise = n.getboolean("conv_depthwise")
        self.convt_depthwise = n.getboolean("convt_depthwise")
        self.conv_kernel = csv(n["conv_kernel"])
        self.convt_kernel = csv(n["convt_kernel"])
        self.conv_kernel_inp = csv(n["conv_kernel_inp"])
        self.emb_hidden_dim = n.getint("emb_hidden_dim")
        self.emb_num_layers = n.getint("emb_num_layers")
        self.emb_gru_skip_enc = n.get("emb_gru_skip_enc", "none")
        self.emb_gru_skip = n["emb_gru_skip"]
        self.df_hidden_dim = n.getint("df_hidden_dim")
        self.df_gru_skip = n["df_gru_skip"]
        self.df_pathway_kernel_size_t = n.getint("df_pathway_kernel_size_t")
        self.enc_concat = n.getboolean("enc_concat")
        self.df_num_layers = n.getint("df_num_layers")
        self.df_n_iter = n.getint("df_n_iter")
        self.lin_groups = n.getint("linear_groups")
        self.enc_lin_groups = n.getint("enc_linear_groups")
        self.mask_pf = n.getboolean("mask_pf")
        self.pf_beta = 0.02
        self.lsnr_dropout = n.getboolean("lsnr_dropout")


def extract(path, names):
    tree = ast.parse(open(path).read())
    keep = [n for n in tree.body if getattr(n, "name", None) in names]
    assert len(keep) == len(names), (path, [n.name for n in keep])
    return ast.unparse(ast.Module(body=keep, type_ignores=[]))


log = types.SimpleNamespace(warning=print, info=print)
ns = dict(
    torch=torch, nn=nn, Tensor=Tensor, F=torch.nn.functional, init=init, Parameter=Parameter,
    math=math, np=np, partial=partial, Final=Final, List=List, Optional=Optional, Tuple=Tuple,
    Union=Union, Callable=Callable, Iterable=Iterable, ModelParams=ModelParams, logger=log,
    PI=3.1415926535897932384626433, as_complex=torch.view_as_complex,
)
mf = dict(ns)
exec(extract(f"{SRC}/multiframe.py", ["MultiFrameModule", "df", "DF"]), mf)
ns["MF"] = types.SimpleNamespace(DF=mf["DF"])
exec(extract(f"{SRC}/modules.py", ["Conv2dNormAct", "ConvTranspose2dNormAct", "Mask",
                                  "SqueezedGRU_S", "GroupedLinearEinsum"]), ns)
exec(extract(f"{SRC}/deepfilternet3.py", ["Add", "Concat", "Encoder", "ErbDecoder",
                                         "DfOutputReshapeMF", "DfDecoder", "DfNet"]), ns)

# --- libDF (lib.rs) en numpy f32 -------------------------------------------------------
p = ModelParams()
sd = torch.load(CKPT, weights_only=True, map_location="cpu")
erb_fb = sd["erb_fb"]
widths = (erb_fb > 0).sum(0).numpy().astype(int)


def freq2erb(f):
    return 9.265 * math.log1p(f / (24.7 * 9.265))


def erb2freq(e):
    return 24.7 * 9.265 * (math.exp(e / 9.265) - 1.0)


def libdf_erb_widths(sr, fft, nb, minf):
    lo, hi = freq2erb(0.0), freq2erb(sr // 2)
    step, fw = (hi - lo) / nb, sr / fft
    erb, prev, over = [], 0, 0
    for i in range(1, nb + 1):
        fb = int(round(erb2freq(lo + i * step) / fw))
        nf = fb - prev - over
        if nf < minf:
            over, nf = minf - nf, minf
        else:
            over = 0
        erb.append(nf)
        prev = fb
    erb[-1] += 1
    erb[-1] -= sum(erb) - (fft // 2 + 1)
    return np.array(erb)


assert (libdf_erb_widths(p.sr, p.fft_size, p.nb_erb, p.min_nb_freqs) == widths).all(), widths

N, H = p.fft_size, p.hop_size
i = np.arange(N)
window = np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * (i + 0.5) / (N // 2)) ** 2).astype(np.float32)
wnorm = np.float32(1.0 / (N * N / (2 * H)))
alpha = np.float32(0.99)  # get_norm_alpha : round(exp(-hop/sr/tau), 3)
assert round(math.exp(-H / p.sr / p.norm_tau), 3) == 0.99


def analysis(x):
    T = len(x) // H
    mem = np.zeros(N - H, np.float32)
    out = np.zeros((T, N // 2 + 1), np.complex64)
    for t in range(T):
        frame = x[t * H:(t + 1) * H]
        buf = np.concatenate([mem, frame]) * window
        mem = frame.copy()
        out[t] = (np.fft.rfft(buf.astype(np.float64)) * wnorm).astype(np.complex64)
    return out


def synthesis(spec):
    T = spec.shape[0]
    mem = np.zeros(N - H, np.float32)
    out = np.zeros(T * H, np.float32)
    for t in range(T):
        s = spec[t].astype(np.complex128).copy()
        s[0] = s[0].real
        s[-1] = s[-1].real
        x = (np.fft.irfft(s, N) * N).astype(np.float32) * window
        out[t * H:(t + 1) * H] = x[:H] + mem
        mem = x[H:]
    return out


def features(spec):
    T = spec.shape[0]
    pw = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    b = np.cumsum(np.concatenate([[0], widths]))
    erb = np.stack([pw[:, b[k]:b[k + 1]].mean(1) for k in range(len(widths))], 1)
    erb = (np.log10(erb + 1e-10) * 10).astype(np.float32)
    s = np.linspace(-60, -90, len(widths)).astype(np.float32)
    for t in range(T):
        s = erb[t] * (1 - alpha) + s * alpha
        erb[t] = (erb[t] - s) / 40.0
    u = np.linspace(0.001, 0.0001, p.nb_df).astype(np.float32)
    cf = spec[:, :p.nb_df].copy()
    for t in range(T):
        u = np.abs(cf[t]) * (1 - alpha) + u * alpha
        cf[t] = cf[t] / np.sqrt(u)
    return erb, cf


# --- enhance() officiel ---------------------------------------------------------------
model = ns["DfNet"](sd["erb_fb"], sd["mask.erb_inv_fb"])
missing = model.load_state_dict(sd, strict=True)
model.eval()

audio, sr = sf.read(WAV, dtype="float32")
assert sr == 48000 and audio.ndim == 1
orig = len(audio)
x = np.concatenate([audio, np.zeros(N, np.float32)])
spec = analysis(x)
erb, cf = features(spec)
spec_t = torch.view_as_real(torch.from_numpy(spec))[None, None]
erb_t = torch.from_numpy(erb)[None, None]
cf_t = torch.view_as_real(torch.from_numpy(cf))[None, None]
with torch.no_grad():
    enh, m, lsnr, coefs = model(spec_t.clone(), erb_t, cf_t)
enh_c = torch.view_as_complex(enh[0, 0].contiguous()).numpy()
y = synthesis(enh_c)[N - H:orig + N - H]
sf.write(f"{OUT}/ref_out.wav", y, 48000, subtype="FLOAT")
save_file({
    "audio": torch.from_numpy(audio), "erb_feat": erb_t[0, 0].contiguous(),
    "spec_feat": cf_t[0, 0].contiguous(), "mask": m[0, 0].contiguous(),
    "lsnr": lsnr[0, :, 0].contiguous(), "spec_out": enh[0, 0].contiguous(),
    "out": torch.from_numpy(y),
}, f"{OUT}/golden.safetensors")
save_file({k: v.float().contiguous() for k, v in sd.items() if v.dtype.is_floating_point},
          f"{OUT}/dfn3.safetensors")
print("ok", orig, "frames", spec.shape[0], "rms in/out", audio.std(), y.std())
