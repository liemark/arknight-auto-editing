"""Standalone audio matching lab kit: no pipes, file-redirected child stdio."""
import os, json, wave, shutil, subprocess
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"
# 本机(2026-09-16 实测)没装 ffmpeg, 而 render_timeline.py / e2e_verify.py 只用它把 wav 转 mp3 方便试听.
# 以前会 FileNotFoundError [WinError 2] 硬崩在 subprocess, 把前面已经算好的结果全丢掉.
# 这里改成: 明确警告一次 + 跳过, wav 照常产出(任何播放器都能放).
HAVE_FFMPEG = shutil.which("ffmpeg") is not None
_ff_warned = False


def ff_available():
    return HAVE_FFMPEG


def run(cmd, tag="run", cwd=None):
    o = os.path.join(_HERE, "_%s.out" % tag)
    e = os.path.join(_HERE, "_%s.err" % tag)
    with open(o, "wb") as fo, open(e, "wb") as fe:
        p = subprocess.run(cmd, stdout=fo, stderr=fe, cwd=cwd)
    if p.returncode != 0:
        try:
            msg = open(e, "r", errors="replace").read()[-2500:]
        except Exception:
            msg = ""
        raise RuntimeError("%s exit %d\n%s" % (tag, p.returncode, msg))
    return o


def ff(args, tag="ff"):
    global _ff_warned
    if not HAVE_FFMPEG:
        if not _ff_warned:
            print("[alab] 本机没有 ffmpeg -> 跳过全部 mp3 转码, 只产出 wav (wav 任何播放器都能放). "
                  "想顺带出 mp3: 装 ffmpeg 并加进 PATH.", flush=True)
            _ff_warned = True
        return None
    return run([FFMPEG, "-hide_banner", "-nostdin", "-y"] + list(args), tag)


def probe(path):
    o = run([FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", path], "probe")
    with open(o, "r", encoding="utf-8") as f:
        return json.load(f)


def wav_read(path, mono=True):
    with wave.open(path, "rb") as w:
        ch, sr, sw, n = w.getnchannels(), w.getframerate(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    a = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if sw == 2:
        a /= 32768.0
    elif sw == 4:
        a /= 2147483648.0
    elif sw == 1:
        a = (a - 0.0) / 128.0
    if ch > 1:
        a = a.reshape(-1, ch)
        if mono:
            a = a.mean(axis=1)
    return a, sr


def wav_write(path, x, sr):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    i = np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(x.shape[1]); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(i.tobytes())
    return path


def ncc(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def xcorr(x, m, chunk=1 << 21):
    """Normalized zero-mean cross-correlation, valid lags. out[i] = ncc(x[i:i+L], m)."""
    x = np.asarray(x, dtype=np.float64); L = len(m)
    n = len(x)
    if n < L:
        return np.zeros(0)
    csum = np.concatenate(([0.0], np.cumsum(x)))
    csum2 = np.concatenate(([0.0], np.cumsum(x * x)))
    m0 = np.asarray(m, dtype=np.float64); m0 = m0 - m0.mean()
    mn = np.sqrt((m0 * m0).sum())
    mr = m0[::-1].copy()
    out = np.empty(n - L + 1, dtype=np.float64)
    for s in range(0, n - L + 1, chunk):
        e = min(s + chunk, n - L + 1)
        seg = x[s:e + L - 1]
        nf = 1 << (int(np.ceil(np.log2(len(seg)))) + 1)
        c = np.fft.irfft(np.fft.rfft(seg, nf) * np.fft.rfft(mr, nf), nf)
        out[s:e] = c[L - 1: e - s + L - 1]
    s1 = csum[L:] - csum[:-L]
    s2 = csum2[L:] - csum2[:-L]
    var = np.maximum(s2 - s1 * s1 / L, 1e-12)
    return out / (np.sqrt(var) * mn)


def bandpass(x, sr, lo, hi):
    x = np.asarray(x, dtype=np.float64); n = len(x)
    nf = 1 << int(np.ceil(np.log2(max(n, 2))))
    X = np.fft.rfft(x, nf)
    f = np.fft.rfftfreq(nf, 1.0 / sr)
    X[(f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(X, nf)[:n]


def frame_rms(x, sr, frame=0.010, hop=0.005):
    fl = max(1, int(round(frame * sr))); hp = max(1, int(round(hop * sr)))
    n = len(x)
    nf = 1 + (n - fl) // hp if n >= fl else 0
    if nf <= 0:
        return np.zeros(0), hop / sr
    xd = np.asarray(x, dtype=np.float64)
    c2 = np.concatenate(([0.0], np.cumsum(xd * xd)))
    idx = np.arange(nf) * hp
    e = (c2[idx + fl] - c2[idx]) / fl
    return np.sqrt(np.maximum(e, 0.0)), hp / sr


def band_env(x, sr, lo=500.0, hi=4000.0, frame=0.010, hop=0.005):
    return frame_rms(bandpass(x, sr, lo, hi), sr, frame, hop)


def find_peaks(score, thr, refr, max_n=200000):
    score = np.asarray(score, dtype=np.float64)
    cand = np.where(score >= thr)[0]
    if cand.size == 0:
        return np.array([], dtype=int), np.array([])
    order = cand[np.argsort(-score[cand], kind="stable")]
    taken = np.zeros(len(score), dtype=bool)
    out = []
    for i in order:
        a = max(0, i - refr); b = min(len(score), i + refr + 1)
        if taken[a:b].any():
            continue
        taken[i] = True
        out.append(int(i))
        if len(out) >= max_n:
            break
    out = np.array(sorted(out), dtype=int)
    return out, score[out]


def adapt_thr(score, k=6.0):
    s = np.asarray(score, dtype=np.float64)
    med = float(np.median(s))
    mad = float(np.median(np.abs(s - med))) * 1.4826
    return med + k * (mad if mad > 1e-9 else 1e-6)


def nnls(A, b, kmax=12):
    """Exact non-negative least squares by active-set enumeration (exact for small n)."""
    A = np.asarray(A, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    m, n = A.shape
    if n > kmax:
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        return np.maximum(x, 0.0)
    best = None
    for mask in range(1, 1 << n):
        cols = [i for i in range(n) if (mask >> i) & 1]
        As = A[:, cols]
        try:
            z = np.linalg.lstsq(As, b, rcond=None)[0]
        except Exception:
            continue
        if (z < -1e-9).any():
            continue
        r = b - As @ z
        c = float(r @ r)
        if best is None or c < best[0]:
            best = (c, cols, z)
    if best is None:
        return np.zeros(n)
    x = np.zeros(n); x[best[1]] = best[2]
    return x


def gain_estimates(obs, tmpls, onsets, span=None):
    """Independent LS gains: g_i = <obs, t_i>/||t_i||^2 (no cross-talk removal)."""
    n = len(obs)
    out = []
    for t in tmpls:
        L = len(t)
        tc = t - t.mean()
        num = 0.0
        for o in onsets:
            if span is not None and o != span:
                continue
            num = float(np.dot(obs[o:o + L], tc))
        out.append(num / float(tc @ tc))
    return np.array(out)
