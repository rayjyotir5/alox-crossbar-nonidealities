"""
AlOx memristor: device-level model -> 64x64 crossbar aggregate behaviour
========================================================================

A valence-change (filamentary) model of a Pt / Al2O3 / Ti resistive-switching
cell, then 4096 of them assembled into a crossbar so we can watch how
*device-to-device variability* shapes the aggregate behaviour that actually
matters for in-memory computing.

Physics, briefly
-----------------
A single internal state variable  w in [0, 1]  tracks how complete the
conductive oxygen-vacancy filament is:

    w -> 1 : filament bridges the oxide      -> Low  Resistance State (LRS)
    w -> 0 : filament ruptured, insulating   -> High Resistance State (HRS)

Conduction blends an ohmic LRS branch with a sinh-nonlinear HRS branch:

    I(V, w) = w * (V / Ron)  +  (1 - w) * (V0 / Roff) * sinh(V / V0)

The state only moves when the bias exceeds a threshold (the SET / RESET
events). Because w changes at the loop corners but stays frozen along the
sides, a single voltage maps to two resistances -> *pinched hysteresis*.
Because w survives at V = 0, the device *remembers* its state -> *memory*.

Run:  python3 alox_crossbar.py
Outputs three PNGs in the working directory + a printed summary.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless: write PNGs, no display needed
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. DEVICE MODEL  (fully vectorised: state w has arbitrary shape)
# ----------------------------------------------------------------------
class AloxMemristor:
    """A population of AlOx cells sharing one update rule.

    Every parameter may be a scalar (identical devices) or an array
    broadcastable to `shape` (variable devices -> the crossbar case).
    """

    def __init__(self, shape=(1,), Ron=2.0e3, Roff=2.0e5, Vset=0.80,
                 Vreset=0.70, kset=80.0, kreset=80.0, V0=0.45, p=4,
                 w0=0.02, noise=0.0, rng=None):
        self.shape = shape
        self.rng = rng if rng is not None else np.random.default_rng(0)
        b = lambda x: np.broadcast_to(np.asarray(x, float), shape).astype(float).copy()
        self.Ron, self.Roff = b(Ron), b(Roff)
        self.Vset, self.Vreset = b(Vset), b(Vreset)
        self.kset, self.kreset = b(kset), b(kreset)
        self.V0 = b(V0)
        self.p = p
        self.noise = noise                 # cycle-to-cycle switching noise (frac.)
        self.w = b(w0)

    # --- conduction: current at applied bias V (read OR write voltage) ---
    def current(self, V):
        V = np.asarray(V, float)
        i_lrs = V / self.Ron
        i_hrs = (self.V0 / self.Roff) * np.sinh(V / self.V0)
        return self.w * i_lrs + (1.0 - self.w) * i_hrs

    # small-signal read conductance at a benign (sub-threshold) voltage
    def read_conductance(self, Vread=0.2):
        return self.current(Vread) / Vread

    # --- state update: ion drift only above SET / RESET thresholds ---
    def step(self, V, dt):
        V = np.broadcast_to(np.asarray(V, float), self.shape)
        dw = np.zeros(self.shape)
        if self.noise:                     # multiplicative cycle-to-cycle noise
            jit = 1.0 + self.noise * self.rng.standard_normal(self.shape)
        else:
            jit = 1.0
        set_mask = V > self.Vset                       # grow filament
        rst_mask = V < -self.Vreset                    # rupture filament
        win_set = 1.0 - self.w ** self.p               # ease as w -> 1
        win_rst = 1.0 - (1.0 - self.w) ** self.p       # ease as w -> 0
        dw = np.where(set_mask,  self.kset  * (V - self.Vset)  * win_set * jit, dw)
        dw = np.where(rst_mask, -self.kreset * (-V - self.Vreset) * win_rst * jit, dw)
        self.w = np.clip(self.w + dw * dt, 1e-3, 1.0 - 1e-3)


# ----------------------------------------------------------------------
# 2. SINGLE-DEVICE SWEEP  (the pinched hysteresis loop)
# ----------------------------------------------------------------------
def sweep_iv(dev, amp=1.5, freq=1.0, n=4000, reset_first=True):
    """Quasi-static triangular V sweep 0->+amp->-amp->0; returns (V, I, w)."""
    if reset_first:
        dev.w[...] = 0.02
    T = 1.0 / freq
    t = np.linspace(0.0, T, n)
    # triangular wave, one full cycle, peaks at +/- amp
    x = (t / T)
    tri = np.where(x < 0.25, 4 * x,
          np.where(x < 0.75, 2 - 4 * x, 4 * x - 4))
    V = amp * tri
    dt = T / n
    I = np.empty(n)
    W = np.empty(n)
    for k in range(n):
        I[k] = dev.current(V[k])[0]
        dev.step(V[k], dt)
        W[k] = dev.w[0]
    return V, I, W


def figure_single_device():
    rng = np.random.default_rng(1)
    dev = AloxMemristor(shape=(1,), rng=rng)
    V, I, W = sweep_iv(dev, amp=1.5, freq=1.0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    fig.suptitle("AlOx single device  —  Pt / Al$_2$O$_3$ / Ti", fontsize=13)

    # (a) pinched hysteresis I-V
    ax[0].plot(V, I * 1e6, color="#d9821a", lw=1.8)
    ax[0].axvline(0.80, ls="--", c="grey", lw=.8); ax[0].axvline(-0.70, ls="--", c="grey", lw=.8)
    ax[0].text(0.80, ax[0].get_ylim()[1]*0.9, " Vset", color="grey", fontsize=8)
    ax[0].text(-0.70, ax[0].get_ylim()[1]*0.9, "Vreset ", color="grey", fontsize=8, ha="right")
    ax[0].set(xlabel="Voltage (V)", ylabel="Current (µA)",
              title="(a) Pinched hysteresis loop")
    ax[0].axhline(0, c="k", lw=.5); ax[0].axvline(0, c="k", lw=.5)
    ax[0].grid(alpha=.25)

    # (b) state variable vs voltage (shows SET/RESET switching)
    ax[1].plot(V, W, color="#2a7fb8", lw=1.6)
    ax[1].set(xlabel="Voltage (V)", ylabel="state  w  (filament completeness)",
              title="(b) Internal state w over the sweep", ylim=(-0.03, 1.03))
    ax[1].grid(alpha=.25)

    # (c) frequency dependence: loop collapses when ions can't keep up.
    # Decade-spaced rates: w reaches ~1.0 / 0.65 / 0.09 respectively, so the
    # loop area shrinks monotonically toward a single (memory-less) line.
    for f, c in [(1.0, "#08415c"), (10.0, "#d9821a"), (100.0, "#c1121f")]:
        d2 = AloxMemristor(shape=(1,), rng=np.random.default_rng(1))
        Vf, If, _ = sweep_iv(d2, amp=1.5, freq=f)
        ax[2].plot(Vf, If * 1e6, lw=1.5, color=c, label=f"{f:.0f} Hz")
    ax[2].set(xlabel="Voltage (V)", ylabel="Current (µA)",
              title="(c) Faster sweep → loop collapses")
    ax[2].axhline(0, c="k", lw=.5); ax[2].axvline(0, c="k", lw=.5)
    ax[2].legend(title="sweep rate", fontsize=8); ax[2].grid(alpha=.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("fig1_single_device.png", dpi=130)
    plt.close(fig)
    print("  wrote fig1_single_device.png")


# ----------------------------------------------------------------------
# 3. 64x64 CROSSBAR  (device-to-device variability)
# ----------------------------------------------------------------------
def make_variable_array(N=64, rng=None):
    """4096 cells with realistic parameter spread (lognormal R, normal Vth)."""
    rng = rng or np.random.default_rng(42)
    shape = (N, N)
    # log-normal so resistances stay positive; HRS spreads more than LRS (real)
    Ron  = 2.0e3 * np.exp(0.15 * rng.standard_normal(shape))   # ~15 % spread
    Roff = 2.0e5 * np.exp(0.30 * rng.standard_normal(shape))   # ~30 % spread
    Vset    = np.clip(0.80 + 0.08 * rng.standard_normal(shape), 0.4, 1.4)
    Vreset  = np.clip(0.70 + 0.08 * rng.standard_normal(shape), 0.4, 1.4)
    return AloxMemristor(shape=shape, Ron=Ron, Roff=Roff, Vset=Vset,
                         Vreset=Vreset, w0=0.02, noise=0.05, rng=rng)


def pulse_all(dev, V, dt=2e-4, n=20):
    """Apply n identical voltage pulses to the whole array (SET or RESET)."""
    for _ in range(n):
        dev.step(V, dt)


def figure_array_variability():
    rng = np.random.default_rng(42)
    arr = make_variable_array(rng=rng)

    # drive the whole array fully OFF then read HRS distribution
    pulse_all(arr, -1.4, n=40)
    g_hrs = arr.read_conductance().ravel()
    # drive the whole array fully ON then read LRS distribution
    pulse_all(arr, +1.4, n=40)
    g_lrs = arr.read_conductance().ravel()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.suptitle("64×64 array (4096 cells)  —  device-to-device variability", fontsize=13)

    # (a) conductance distributions of the two memory states
    ax[0].hist(g_hrs * 1e6, bins=60, color="#2a7fb8", alpha=.8, label="HRS (logic 0)")
    ax[0].hist(g_lrs * 1e6, bins=60, color="#d9821a", alpha=.8, label="LRS (logic 1)")
    ax[0].set(xlabel="read conductance (µS)", ylabel="number of cells",
              title="(a) State separation across the array", xscale="log")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.25)
    margin = g_lrs.min() / g_hrs.max()
    ax[0].text(0.02, 0.95, f"on/off read margin ≈ {margin:.0f}×",
               transform=ax[0].transAxes, fontsize=9, va="top")

    # (b) variability cloud: I-V loops of 60 random cells overlaid
    idx = rng.choice(64 * 64, 60, replace=False)
    for j in idx:
        r, c = divmod(j, 64)
        d = AloxMemristor(shape=(1,), Ron=arr.Ron[r, c], Roff=arr.Roff[r, c],
                          Vset=arr.Vset[r, c], Vreset=arr.Vreset[r, c],
                          rng=np.random.default_rng(int(j)))
        Vj, Ij, _ = sweep_iv(d, amp=1.5, freq=1.0)
        ax[1].plot(Vj, Ij * 1e6, color="#d9821a", lw=.6, alpha=.25)
    ax[1].axhline(0, c="k", lw=.5); ax[1].axvline(0, c="k", lw=.5)
    ax[1].set(xlabel="Voltage (V)", ylabel="Current (µA)",
              title="(b) 60 cells overlaid — no two loops identical")
    ax[1].grid(alpha=.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("fig2_array_variability.png", dpi=130)
    plt.close(fig)
    print("  wrote fig2_array_variability.png")
    print(f"    HRS conductance: {g_hrs.mean()*1e6:7.3f} µS  (sigma/mean {g_hrs.std()/g_hrs.mean():.2f})")
    print(f"    LRS conductance: {g_lrs.mean()*1e6:7.3f} µS  (sigma/mean {g_lrs.std()/g_lrs.mean():.2f})")
    print(f"    on/off read margin: {margin:.0f}x")


# ----------------------------------------------------------------------
# 4. CROSSBAR AS COMPUTE: analog programming + vector-matrix multiply
# ----------------------------------------------------------------------
def make_target_image(N=64):
    """Render 'AlOx' to an NxN grayscale array in [0,1] (the target pattern)."""
    fig = plt.figure(figsize=(1, 1), dpi=N)
    fig.patch.set_facecolor("black")                 # capture black, not white
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("black")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, 0.5, "AlOx", color="white", ha="center", va="center",
            fontsize=15, weight="bold", family="monospace")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].mean(2) / 255.0
    plt.close(fig)
    return buf[:N, :N]             # canvas buffer is already row-0-at-top


def program_to_conductance(dev, G_target, Vread=0.2, n_iter=60,
                           Vset_prog=1.25, Vreset_prog=1.20, dt_p=8e-4):
    """Closed-loop program-verify: nudge each cell toward its target G.

    Each round: read every cell, SET-pulse the ones below target and
    RESET-pulse the ones above. Cycle-to-cycle noise + finite step size
    leave a residual write error — exactly the real-hardware behaviour.
    """
    for _ in range(n_iter):
        G = dev.read_conductance(Vread)
        low  = G < G_target * 0.98
        high = G > G_target * 1.02
        dev.step(np.where(low,  Vset_prog, 0.0), dt_p)     # raise w where low
        dev.step(np.where(high, -Vreset_prog, 0.0), dt_p)  # lower w where high


def figure_crossbar_compute():
    rng = np.random.default_rng(7)
    arr = make_variable_array(rng=rng)

    # --- map an image to a target conductance matrix (the stored "weights") ---
    img = make_target_image(64)
    # weight window = the conductance range EVERY device can reach despite
    # variability (above the worst HRS floor, below the worst LRS ceiling).
    g_lo, g_hi = 20e-6, 300e-6                   # 20..300 µS  (S)
    G_target = g_lo + img * (g_hi - g_lo)        # 64x64 desired conductances

    arr.w[...] = 0.02                            # start fully RESET (HRS)
    program_to_conductance(arr, G_target)
    G_actual = arr.read_conductance()            # what the array actually holds

    # recover the stored image from read conductances (normalise back to [0,1])
    img_read = np.clip((G_actual - g_lo) / (g_hi - g_lo), 0, 1)
    write_err = (G_actual - G_target) / (g_hi - g_lo)   # error in image units

    # --- in-memory vector-matrix multiply (Ohm + Kirchhoff, one shot) ---
    #     bitline current  I_j = sum_i  V_i * G_ij   (ideal 1T1R, no IR drop)
    Vin = rng.uniform(0, 0.2, size=64)           # 64 sub-threshold inputs (read)
    I_ideal  = Vin @ G_target                    # what a perfect array computes
    I_actual = Vin @ G_actual                    # what this variable array gives
    vmm_err = np.abs(I_actual - I_ideal) / np.abs(I_ideal)

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 9))
    fig.suptitle("64×64 crossbar as compute:  store weights, then multiply", fontsize=13)

    ax[0, 0].imshow(img, cmap="magma"); ax[0, 0].axis("off")
    ax[0, 0].set_title("(a) Target weights  G$_{target}$")

    ax[0, 1].imshow(img_read, cmap="magma"); ax[0, 1].axis("off")
    ax[0, 1].set_title("(b) Read back after program-verify")

    im = ax[1, 0].imshow(write_err, cmap="coolwarm", vmin=-0.25, vmax=0.25)
    ax[1, 0].axis("off")
    ax[1, 0].set_title("(c) Write error map (variability + finite steps)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, label="error (image units)")

    ax[1, 1].scatter(I_ideal * 1e6, I_actual * 1e6, s=10, alpha=.6, color="#d9821a")
    lim = [0, max(I_ideal.max(), I_actual.max()) * 1e6 * 1.05]
    ax[1, 1].plot(lim, lim, "k--", lw=.8)
    ax[1, 1].set(xlabel="ideal bitline current (µA)",
                 ylabel="array bitline current (µA)",
                 title="(d) Analog VMM:  ideal vs real",
                 xlim=lim, ylim=lim)
    ax[1, 1].grid(alpha=.25)
    ax[1, 1].text(0.04, 0.92, f"mean |error| = {vmm_err.mean()*100:.1f} %",
                  transform=ax[1, 1].transAxes, fontsize=10, va="top")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("fig3_crossbar_compute.png", dpi=130)
    plt.close(fig)
    print("  wrote fig3_crossbar_compute.png")
    rmse = np.sqrt(np.mean(write_err ** 2))
    print(f"    weight write RMSE: {rmse*100:.1f} % of full scale")
    print(f"    VMM mean relative error: {vmm_err.mean()*100:.1f} %  "
          f"(max {vmm_err.max()*100:.1f} %)")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("AlOx memristor + 64x64 crossbar simulation")
    print("-" * 50)
    print("[1/3] single-device hysteresis & frequency dependence")
    figure_single_device()
    print("[2/3] 64x64 array variability")
    figure_array_variability()
    print("[3/3] crossbar compute: program weights + analog VMM")
    figure_crossbar_compute()
    print("-" * 50)
    print("done. open fig1_single_device.png, fig2_array_variability.png, "
          "fig3_crossbar_compute.png")
