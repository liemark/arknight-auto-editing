
"""增广规格与身份一致性检查。

增广的前提是“变换后的样本仍属于同一类”。同一组变换参数对不同模板的效果差异很大：
能量集中的冲击音可能被 BandPassFilter 整段滤掉，而宽带噪声型模板几乎不受影响；
变速则会破坏固定的时间对齐。因此每条变体都要与它的干净模板逐条比对一致性，
不达标就换用 SAFE 规格重做。

SAFE       只保留与时间对齐相容的失真：增益 / 增益包络 / 参量 EQ / 噪声 / MP3 / 限幅 / 响度。
SPEC_TAMED SAFE 加 带通/削波/bitcrush/混叠/变速 的温和版本。
SPEC_FULL  最激进的一档，必须配合 QC 使用。
"""
import numpy as np

# original (broken) spec, kept for A/B
SPEC_FULL = [
    ("Gain", dict(min_gain_db=-14.0, max_gain_db=6.0, p=1.0)),
    ("GainTransition", dict(min_gain_db=-18.0, max_gain_db=0.0, min_duration=0.05, max_duration=0.8, p=0.35)),
    ("BandPassFilter", dict(min_center_freq=80, max_center_freq=5500, p=0.75)),
    ("SevenBandParametricEQ", dict(min_gain_db=-6.0, max_gain_db=6.0, p=0.5)),
    ("HighShelfFilter", dict(min_gain_db=-12.0, max_gain_db=6.0, p=0.4)),
    ("LowShelfFilter", dict(min_gain_db=-12.0, max_gain_db=6.0, p=0.4)),
    ("ClippingDistortion", dict(min_percentile_threshold=3, max_percentile_threshold=35, p=0.25)),
    ("TanhDistortion", dict(min_distortion=0.05, max_distortion=0.6, p=0.25)),
    ("BitCrush", dict(min_bit_depth=8, max_bit_depth=14, p=0.10)),
    ("Aliasing", dict(min_sample_rate=6000, max_sample_rate=14000, p=0.30)),
    ("AddGaussianSNR", dict(min_snr_db=12.0, max_snr_db=45.0, p=0.35)),
    ("AddColorNoise", dict(min_snr_db=15.0, max_snr_db=45.0, p=0.25)),
    ("TimeStretch", dict(min_rate=0.94, max_rate=1.06, p=0.15)),
    ("Mp3Compression", dict(min_bitrate=24, max_bitrate=96, p=0.25)),
    ("Limiter", dict(min_threshold_db=-18.0, max_threshold_db=-2.0, min_release=0.05, max_release=0.4, p=0.35)),
    ("LoudnessNormalization", dict(min_lufs=-31.0, max_lufs=-13.0, p=0.35)),
    ("Normalize", dict(p=1.0)),
]

# "safe" core: only jointly-alignment-preserving distortion
SAFE = [
    ("Gain", dict(min_gain_db=-12.0, max_gain_db=6.0, p=1.0)),
    ("GainTransition", dict(min_gain_db=-18.0, max_gain_db=0.0, min_duration=0.05, max_duration=0.8, p=0.30)),
    ("SevenBandParametricEQ", dict(min_gain_db=-6.0, max_gain_db=6.0, p=0.60)),
    ("HighShelfFilter", dict(min_gain_db=-12.0, max_gain_db=6.0, p=0.40)),
    ("LowShelfFilter", dict(min_gain_db=-12.0, max_gain_db=6.0, p=0.40)),
    ("AddGaussianSNR", dict(min_snr_db=20.0, max_snr_db=45.0, p=0.30)),
    ("AddColorNoise", dict(min_snr_db=25.0, max_snr_db=45.0, p=0.20)),
    ("Mp3Compression", dict(min_bitrate=48, max_bitrate=128, p=0.25)),
    ("Limiter", dict(min_threshold_db=-12.0, max_threshold_db=-2.0, min_release=0.05, max_release=0.4, p=0.30)),
    ("LoudnessNormalization", dict(min_lufs=-31.0, max_lufs=-13.0, p=0.30)),
    ("Normalize", dict(p=1.0)),
]

# tamed full: SAFE + gentler versions of the "flavour" transforms
SPEC_TAMED = SAFE + [
    ("BandPassFilter", dict(min_center_freq=300, max_center_freq=3000,
                            min_bandwidth_fraction=1.0, max_bandwidth_fraction=1.9,
                            min_rolloff=12, max_rolloff=24, p=0.30)),
    ("ClippingDistortion", dict(min_percentile_threshold=10, max_percentile_threshold=40, p=0.15)),
    ("TanhDistortion", dict(min_distortion=0.02, max_distortion=0.30, p=0.15)),
    ("BitCrush", dict(min_bit_depth=10, max_bit_depth=14, p=0.08)),
    ("Aliasing", dict(min_sample_rate=11000, max_sample_rate=15000, p=0.10)),
]

QC_NCC = 0.60

# 部分变换有最小长度要求：LoudnessNormalization（pyloudnorm）需要 >= 400 ms，
# 其余变换 100 ms 即可。长度不足时必须显式剔除该变换 —— 若让它抛异常再由调用方的
# except 兜住，那条变体会退化成“原样复制”，而它的 NCC = 1.0 会被判为合格，
# 等于悄无声息地失去增广。
MIN_SAMPLES = {"LoudnessNormalization": 6400}      # 400 ms @ 16 kHz


def _compose(spec, n_samples=None):
    import audiomentations as A
    ts, skipped = [], []
    for name, kw in spec:
        need = MIN_SAMPLES.get(name, 0)
        if n_samples is not None and n_samples < need:
            skipped.append("%s(需 %d 采样, 只有 %d)" % (name, need, n_samples))
            continue
        try:
            ts.append(getattr(A, name)(**kw))
        except Exception as e:
            skipped.append("%s(%s)" % (name, str(e)[:40]))
    return A.Compose(ts), skipped


def build(profile="tamed", n_samples=None):
    """n_samples 给定时，自动过滤掉在这个长度上跑不动的变换（见 MIN_SAMPLES）。"""
    if profile == "full":
        return _compose(SPEC_FULL, n_samples)
    if profile == "safe":
        return _compose(SAFE, n_samples)
    return _compose(SPEC_TAMED, n_samples)


def ncc(a, b):
    """Zero-mean normalized correlation over the common prefix (alignment preserved)."""
    n = min(len(a), len(b))
    if n < 32:
        return 0.0
    a = np.asarray(a[:n], dtype=np.float64); b = np.asarray(b[:n], dtype=np.float64)
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 1e-12 else 0.0
