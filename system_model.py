"""
Enhanced crossbar system model for inference-only mapping of a real network.
Adds the system realism a hardware reviewer expects (point 5): layer tiling
(im2col for conv), a 1T1R access-transistor option, ADC/DAC quantization, and
peripheral read noise, on top of the calibrated Ag2O device and the full IR-drop
nodal solve. Exposes run_config(cfg) -> (task accuracy, cheap MVM-error feature)
so the operating envelope can be built on REAL task accuracy (points 2,3).
"""
import numpy as np
from crossbar_advanced import CrossbarIR, VREAD

# device calibrated to Keshari et al. (see keshari_phase_energy.py)
G_LO, G_HI = 3.6e-6, 22.3e-6
N_LEVELS = 100
VAR_SIGMA = 0.02
R_T = 1.0e4                       # 1T1R access-transistor on-resistance (ohm)
TOL = 0.08                        # steering within-tolerance threshold (task metric)


def _quant(a, lo, hi, bits):
    if bits is None or bits <= 0:
        return a
    lv = np.round((a - lo) / (hi - lo + 1e-12) * (2 ** bits - 1))
    return lo + np.clip(lv, 0, 2 ** bits - 1) / (2 ** bits - 1) * (hi - lo)


def deploy(W, rng):
    half, mid = (G_HI - G_LO) / 2, (G_HI + G_LO) / 2
    wmax = np.abs(W).max() + 1e-12
    Wn = W / wmax
    q = lambda G: _quant(G, G_LO, G_HI, int(np.log2(N_LEVELS)) + 1)
    Gp, Gm = q(mid + Wn * half), q(mid - Wn * half)
    Gp = np.clip(Gp * np.exp(VAR_SIGMA * rng.standard_normal(Gp.shape)), G_LO, G_HI)
    Gm = np.clip(Gm * np.exp(VAR_SIGMA * rng.standard_normal(Gm.shape)), G_LO, G_HI)
    return Gp, Gm, 2 * half / wmax


def analog_matmul(X, W, cfg, rng):
    """X (K,in) @ W (in,out) on tiled crossbars with the configured non-idealities."""
    K, nin = X.shape
    nout = W.shape[1]
    N = cfg["N"]
    cols_per = max(1, N // 2)                      # differential: 2 phys cols / weight
    Gp, Gm, _ = deploy(W, rng)
    if cfg.get("cell") == "1T1R":                  # access-transistor series resistance
        Gp = 1.0 / (1.0 / Gp + R_T); Gm = 1.0 / (1.0 / Gm + R_T)
    # calibrated readout gain: least-squares map of the (possibly compressed)
    # differential conductance back to the intended weight (what a real readout
    # is calibrated to). Removes systematic bias for both 1R and 1T1R.
    gdiff = Gp - Gm
    a = float(np.sum(W * gdiff) / (np.sum(gdiff * gdiff) + 1e-30))
    Xq = _quant(X, X.min(), X.max(), cfg.get("dac_bits"))
    Vin = Xq * VREAD
    Y = np.zeros((K, nout))
    for r0 in range(0, nin, N):
        rt = slice(r0, min(r0 + N, nin))
        for c0 in range(0, nout, cols_per):
            ct = slice(c0, min(c0 + cols_per, nout))
            Gpc, Gmc = Gp[rt, ct], Gm[rt, ct]
            M = Gpc.shape[1]
            Gmat = np.empty((Gpc.shape[0], 2 * M)); Gmat[:, 0::2] = Gpc; Gmat[:, 1::2] = Gmc
            Vsub = Vin[:, rt]
            if cfg.get("ideal"):
                I = Vsub @ Gmat
            else:
                cb = CrossbarIR(Gmat, cfg["r_w"])
                I = np.empty((K, 2 * M))
                for s in range(0, K, 4000):         # chunk RHS to bound memory
                    I[s:s + 4000] = cb.vmm(Vsub[s:s + 4000])
            diff = I[:, 0::2] - I[:, 1::2]
            if cfg.get("read_noise"):
                diff += cfg["read_noise"] * (np.abs(diff).mean() + 1e-15) * \
                    rng.standard_normal(diff.shape)
            contrib = a * diff / VREAD              # calibrated -> ~ Xq[:,rt] @ W[rt,ct]
            contrib = _quant(contrib, contrib.min(), contrib.max(), cfg.get("adc_bits"))
            Y[:, ct] += contrib
    return Y


def im2col(x, k, s, p):
    B, C, H, Wd = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    Ho = (H + 2 * p - k) // s + 1; Wo = (Wd + 2 * p - k) // s + 1
    cols = np.empty((B, C, k, k, Ho, Wo), x.dtype)
    for i in range(k):
        for j in range(k):
            cols[:, :, i, j] = xp[:, :, i:i + s * Ho:s, j:j + s * Wo:s]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(B * Ho * Wo, C * k * k), Ho, Wo


def conv_analog(x, Wc, bc, s, p, cfg, rng):
    Cout, Cin, k, _ = Wc.shape
    cols, Ho, Wo = im2col(x, k, s, p)
    Wmat = Wc.reshape(Cout, -1).T                  # (Cin*k*k, Cout)
    Y = analog_matmul(cols, Wmat, cfg, rng) + bc
    return np.maximum(0, Y).reshape(x.shape[0], Ho, Wo, Cout).transpose(0, 3, 1, 2)


def infer_pilotnet(w, X, cfg, rng):
    x = conv_analog(X, w["c1w"], w["c1b"], 2, 2, cfg, rng)
    x = conv_analog(x, w["c2w"], w["c2b"], 2, 2, cfg, rng)
    x = conv_analog(x, w["c3w"], w["c3b"], 2, 1, cfg, rng)
    x = x.reshape(x.shape[0], -1)
    x = np.maximum(0, analog_matmul(x, w["f1w"].T, cfg, rng) + w["f1b"])
    return (analog_matmul(x, w["f2w"].T, cfg, rng) + w["f2b"])[:, 0]   # continuous steering


def mvm_error_feature(cfg, rng):
    """Cheap, network-independent MVM error at this config (the envelope feature)."""
    N = cfg["N"]
    G = rng.uniform(G_LO, G_HI, (N, N))
    if cfg.get("cell") == "1T1R":
        G = 1.0 / (1.0 / G + R_T)
    Vin = np.full(N, VREAD)
    ideal = Vin @ G
    I = CrossbarIR(G, cfg["r_w"]).vmm(Vin)[0]
    return float(100 * np.mean(np.abs(I - ideal) / np.abs(ideal)))


def run_config(cfg, npz="robotics_pilotnet.npz", n_test=150, seed=0):
    d = np.load(npz)
    w = {k: d[k] for k in d.files if k not in ("Xtest", "ytest", "acc_sw")}
    X, y = d["Xtest"][:n_test], d["ytest"][:n_test]
    rng = np.random.default_rng(seed)
    pred = infer_pilotnet(w, X, cfg, rng)
    acc = float(np.mean(np.abs(pred - y) < TOL))          # within-tolerance steering
    mae = float(np.mean(np.abs(pred - y)))
    feat = mvm_error_feature(cfg, np.random.default_rng(seed + 1))
    return {"acc": acc * 100, "mae": mae, "mvm_err": feat,
            "acc_sw": float(d["acc_sw"]) * 100, **cfg}


if __name__ == "__main__":     # smoke test: one small config
    import time
    t = time.time()
    r = run_config({"N": 64, "r_w": 2.0, "cell": "1R", "adc_bits": 8,
                    "dac_bits": 6, "read_noise": 0.02, "ideal": False}, n_test=40)
    print(r, "in %.1fs" % (time.time() - t))
