"""
Calibrated-device study: (1) calibrate the conductance model to the published
Keshari et al. (2026) self-rectifying Ag2O 64x64 crossbar, validate against their
MNIST number, (2) the operating-envelope phase diagram over (array size, wire
resistance) for passive vs self-rectifying cells, (3) an energy / TOPS-per-W
model. Run after aihwkit_baseline.py has written mnist_mlp.npz.

Run:  python3 keshari_phase_energy.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from crossbar_advanced import CrossbarIR, VREAD, KB_EV

# ---- device calibrated to Keshari et al. 2026 (Ag2O, self-rectifying) ----
G_LO, G_HI = 3.6e-6, 22.3e-6     # multilevel conductance window at 373 K (S)
N_LEVELS = 16                     # >4-bit multilevel
RECT = 1.0e3                      # self-rectification ratio (>10^3)
T_READ = 100e-9                   # read pulse width (s)
E_ADC = 1.0e-12                   # per-column ADC conversion energy (J), SAR-class
VAR_SIGMA = 0.05                  # programming variability (negligible spatiotemporal)
G_MEAN = 0.5 * (G_LO + G_HI)


def quantize(G):
    lv = np.round((G - G_LO) / (G_HI - G_LO) * (N_LEVELS - 1))
    return G_LO + np.clip(lv, 0, N_LEVELS - 1) / (N_LEVELS - 1) * (G_HI - G_LO)


def deploy(W, sigma=VAR_SIGMA, rng=None):
    """signed weight -> differential (G+,G-) in the Keshari window, 4-bit + noise."""
    half, mid = (G_HI - G_LO) / 2, (G_HI + G_LO) / 2
    wmax = np.abs(W).max()
    Wn = W / wmax
    Gp, Gm = quantize(mid + Wn * half), quantize(mid - Wn * half)
    if sigma and rng is not None:
        Gp = np.clip(Gp * np.exp(sigma * rng.standard_normal(Gp.shape)), G_LO, G_HI)
        Gm = np.clip(Gm * np.exp(sigma * rng.standard_normal(Gm.shape)), G_LO, G_HI)
    return Gp, Gm, 2 * half / wmax        # decode scale: (Gp-Gm)/scale = W


def w_eff(W, sigma, rng):
    Gp, Gm, scale = deploy(W, sigma, rng)
    return (Gp - Gm) / scale


# ======================================================================
# 1. MNIST validation: our calibrated simulator vs software / AIHWKit / Keshari
# ======================================================================
def mnist_validation():
    if not os.path.exists("mnist_mlp.npz"):
        print("[MNIST] mnist_mlp.npz not found (run aihwkit_baseline.py first); skipping.")
        return
    d = np.load("mnist_mlp.npz")
    W1, b1, W2, b2 = d["W1"].T, d["b1"], d["W2"].T, d["b2"]   # torch (out,in)->(in,out)
    X, y, acc_sw = d["Xtest"], d["ytest"], float(d["acc_sw"])
    rng = np.random.default_rng(0)

    def run(sigma):
        accs = []
        for s in range(5):
            r = np.random.default_rng(100 + s)
            a1 = np.maximum(0, X @ w_eff(W1, sigma, r) + b1)
            logits = a1 @ w_eff(W2, sigma, r) + b2
            accs.append(np.mean(logits.argmax(1) == y))
        return 100 * np.mean(accs), 100 * np.std(accs)

    m4, s4 = run(VAR_SIGMA)                  # 4-bit + variability
    print("[MNIST] 784-128-10 MLP, our calibrated Ag2O simulator:")
    print(f"  software (float)                 : {acc_sw*100:.2f}")
    print(f"  our sim (4-bit window + var)     : {m4:.2f} +/- {s4:.2f}")
    print(f"  Keshari et al. 2026 (measured)   : 96.08  (reference baseline)")


# ======================================================================
# 2. Operating-envelope phase diagram: (array size N) x (wire resistance r_w)
# ======================================================================
def mvm_error(G, Vin, r_w, rect=False, iters=2):
    ideal = Vin @ G
    Geff = G.copy()
    cb = CrossbarIR(Geff, r_w)
    if rect:
        N, M = G.shape
        for _ in range(iters):
            x = cb.lu.solve(cb._rhs(Vin))[:, 0]
            Vr = x[:cb.NM].reshape(N, M); Vc = x[cb.NM:2 * cb.NM].reshape(N, M)
            Geff = np.where((Vr - Vc) < 0, G / RECT, G)      # suppress reverse/sneak
            cb = CrossbarIR(Geff, r_w)
    I = cb.vmm(Vin)[0]
    return 100 * np.mean(np.abs(I - ideal) / np.abs(ideal))


def operating_envelopes():
    rng = np.random.default_rng(1)
    # ---- (a) IR envelope: array size N x wire resistance r_w -> MVM error ----
    Ns = [16, 32, 48, 64, 96, 128]
    rws = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    err = np.zeros((len(rws), len(Ns)))
    rect_check = []
    for j, N in enumerate(Ns):
        G = rng.uniform(G_LO, G_HI, (N, N))          # realistic mixed-state map
        Vin = np.full(N, VREAD)
        for i, rw in enumerate(rws):
            err[i, j] = mvm_error(G, Vin, rw, rect=False)
        rect_check.append((mvm_error(G, Vin, 2.0, rect=False),
                           mvm_error(G, Vin, 2.0, rect=True)))
    # self-rectification check (single-ended parallel MVM: all cells forward-biased)
    dp = max(abs(p - r) for p, r in rect_check)
    print(f"[phase] self-rectification effect on read MVM error: max |passive-rect| = {dp:.2f}% "
          f"(rectification suppresses sneak/write, not read-time IR drop)")

    # ---- (b) retention envelope: time x temperature -> trained-MLP accuracy ----
    Ea, tau0 = 1.0, 3.15e8 / np.exp(1.0 / (KB_EV * 298.15))
    temps = np.array([25, 45, 65, 85, 105, 125.])
    times = np.logspace(0, np.log10(3.15e7), 14)     # 1 s .. 1 yr
    accT = np.full((len(temps), len(times)), 50.0)
    if os.path.exists("mnist_mlp.npz"):
        d = np.load("mnist_mlp.npz")
        W1, b1, W2, b2 = d["W1"].T, d["b1"], d["W2"].T, d["b2"]
        X, y = d["Xtest"], d["ytest"]
        r = np.random.default_rng(0)
        Gp1, Gm1, sc1 = deploy(W1, VAR_SIGMA, r)
        Gp2, Gm2, sc2 = deploy(W2, VAR_SIGMA, r)
        for ti, T in enumerate(temps):
            tau = tau0 * np.exp(Ea / (KB_EV * (T + 273.15)))
            for tj, t in enumerate(times):
                f = np.exp(-t / tau)
                drift = lambda G: G_LO + (G - G_LO) * f
                a1 = np.maximum(0, X @ ((drift(Gp1) - drift(Gm1)) / sc1) + b1)
                lo = a1 @ ((drift(Gp2) - drift(Gm2)) / sc2) + b2
                accT[ti, tj] = 100 * np.mean(lo.argmax(1) == y)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    cf = ax[0].contourf(Ns, rws, err, levels=np.linspace(0, 60, 25), cmap="RdYlGn_r", extend="max")
    cs = ax[0].contour(Ns, rws, err, levels=[2, 10, 30], colors="k", linewidths=1.1)
    ax[0].clabel(cs, fmt="%d%%", fontsize=8)
    ax[0].set(xlabel="array size N (N x N)", title="(a) IR-drop envelope (space)")
    ax[0].set_ylabel(r"wire resistance $r_w$ ($\Omega$/cell)")
    ax[0].set_yscale("log"); ax[0].set_yticks(rws); ax[0].set_yticklabels([f"{r:.1f}" for r in rws])
    fig.colorbar(cf, ax=ax[0], fraction=0.046, label="MVM error (%)")

    cf2 = ax[1].contourf(times, temps, accT, levels=np.linspace(50, 96, 24), cmap="RdYlGn", extend="min")
    cs2 = ax[1].contour(times, temps, accT, levels=[90, 93], colors="k", linewidths=1.1)
    ax[1].clabel(cs2, fmt="%d%%", fontsize=8)
    for mark, lab in [(86400, "1d"), (2.6e6, "1mo"), (3.15e7, "1yr")]:
        ax[1].axvline(mark, color="grey", ls=":", lw=.6)
    ax[1].set(xlabel="time since programming (s)", ylabel="temperature ($^\\circ$C)",
              title="(b) Retention envelope (time), MNIST MLP", xscale="log")
    fig.colorbar(cf2, ax=ax[1], fraction=0.046, label="accuracy (%)")
    fig.suptitle("Operating envelopes for the calibrated device: where each remedy is needed "
                 "(IR contours 2/10/30%; retention contours 90/93%)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("phase_diagram.png", dpi=130)
    plt.close(fig)
    print("wrote phase_diagram.png")


# ======================================================================
# 3. Energy model: TOPS/W vs array size, and IR-wasted-energy fraction
# ======================================================================
def energy_model():
    rng = np.random.default_rng(2)
    Ns = np.array([16, 32, 48, 64, 96, 128])
    rw = 2.0
    tops_w, ir_waste = [], []
    digital_ref = 2.0 / 0.5e-12 / 1e12        # ~0.5 pJ/MAC digital -> TOPS/W
    for N in Ns:
        G = rng.uniform(G_LO, G_HI, (N, N))
        Vin = np.full(N, VREAD)
        cb = CrossbarIR(G, rw)
        x = cb.lu.solve(cb._rhs(Vin))[:, 0]
        Vr = x[:cb.NM].reshape(N, N); Vc = x[cb.NM:2 * cb.NM].reshape(N, N)
        Vr0 = Vr[:, 0]
        Psrc = np.sum(cb.gw * (Vin - Vr0) * Vin)           # total power delivered
        Pdev = np.sum(G * (Vr - Vc) ** 2)                  # useful power in devices
        E_total = Psrc * T_READ + N * E_ADC                # + per-column ADC
        ops = 2 * N * N
        tops_w.append(ops / E_total / 1e12)
        ir_waste.append(100 * (Psrc - Pdev) / Psrc)        # fraction lost in wires
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(Ns, tops_w, "o-", color="#2a7fb8", label="crossbar (calibrated)")
    ax[0].axhline(digital_ref, ls="--", color="grey", label="digital MAC (~0.5 pJ)")
    ax[0].set(xlabel="array size N", ylabel="efficiency (TOPS/W)",
              title="(a) Compute efficiency vs array size", yscale="log")
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)
    ax[1].plot(Ns, ir_waste, "o-", color="#c1121f")
    ax[1].set(xlabel="array size N", ylabel="energy lost in wires (%)",
              title=f"(b) IR-drop energy waste ($r_w=2\\,\\Omega$)")
    ax[1].grid(alpha=.3)
    fig.suptitle("Energy model: in-array efficiency and the IR-drop penalty", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("energy_model.png", dpi=130)
    plt.close(fig)
    print(f"[energy] TOPS/W: N=16 {tops_w[0]:.0f}, N=64 {tops_w[3]:.0f}, N=128 {tops_w[-1]:.0f}; "
          f"digital ref {digital_ref:.0f}")
    print(f"[energy] IR waste: N=16 {ir_waste[0]:.1f}%, N=128 {ir_waste[-1]:.1f}%")
    print("wrote energy_model.png")


if __name__ == "__main__":
    print("=" * 60)
    mnist_validation()
    print("-" * 60); operating_envelopes()
    print("-" * 60); energy_model()
    print("=" * 60)
