"""
Crossbar non-idealities: wire IR-drop, retention drift, and an inference study
==============================================================================

Builds on the device model in `alox_crossbar.py` and adds the three effects
that decide whether a crossbar accelerator actually works:

  A. WIRE IR-DROP  — a proper nodal (modified-nodal-analysis) solve of the
     full 2*N*M resistor network. Finite wordline/bitline resistance makes the
     applied voltage sag across the array, so far-corner cells see less drive
     and contribute less current than ideal Ohm+Kirchhoff predicts. The error
     grows with array size — visibly worse at 128x128 than 64x64.

  B. RETENTION / DRIFT — the programmed state w relaxes toward equilibrium with
     an Arrhenius time constant tau(T) = tau0 * exp(Ea / kB T). Stored
     conductance decays over time, faster when hot. This is the physics of a
     memristor *forgetting*.

  C. INFERENCE ACCURACY — YOLO is object detection with millions of trained
     weights and cannot run on (or fit in) a 64x64 array, and its weights can't
     be fetched offline. So we do the thing that answers the real question:
       (1) map YOLO's CORE BUILDING BLOCK — a convolutional layer — onto the
           crossbar (signed weights via a differential G+/G- pair) and measure
           feature-map fidelity, and
       (2) deploy a full linear classifier whose layer FITS the array
           (sklearn digits, 8x8 -> 64 rows) and report end-to-end accuracy
           loss from raw variability, program-verify, IR-drop, and drift.

Run:  python3 crossbar_advanced.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from alox_crossbar import AloxMemristor

KB_EV = 8.617333e-5          # Boltzmann constant, eV/K
VREAD = 0.2                  # sub-threshold read voltage (V)


# ======================================================================
#  Variable-device helpers (device-to-device variability, any shape)
# ======================================================================
def variable_memristor(shape, rng, ron_sig=0.15, roff_sig=0.30, vth_sig=0.08):
    """An AloxMemristor population with realistic parameter spread."""
    Ron  = 2.0e3 * np.exp(ron_sig  * rng.standard_normal(shape))
    Roff = 2.0e5 * np.exp(roff_sig * rng.standard_normal(shape))
    Vset   = np.clip(0.80 + vth_sig * rng.standard_normal(shape), 0.4, 1.4)
    Vreset = np.clip(0.70 + vth_sig * rng.standard_normal(shape), 0.4, 1.4)
    return AloxMemristor(shape=shape, Ron=Ron, Roff=Roff, Vset=Vset,
                         Vreset=Vreset, w0=0.02, noise=0.05, rng=rng)


# nominal (mean-device) conductance <-> state mapping, used for open-loop write
_GON_NOM  = 1.0 / 2.0e3
_GOFF_NOM = (0.45 / 2.0e5) * np.sinh(VREAD / 0.45) / VREAD


def w_for_conductance(G):
    """State w that a *nominal* device needs to read conductance G."""
    w = (G - _GOFF_NOM) / (_GON_NOM - _GOFF_NOM)
    return np.clip(w, 1e-3, 1.0 - 1e-3)


def program_verify(dev, G_target, n_iter=60, Vset_p=1.25, Vreset_p=1.20, dt_p=8e-4):
    """Closed-loop write: read, then SET/RESET-pulse toward each target."""
    for _ in range(n_iter):
        G = dev.read_conductance(VREAD)
        dev.step(np.where(G < G_target * 0.98,  Vset_p,   0.0), dt_p)
        dev.step(np.where(G > G_target * 1.02, -Vreset_p, 0.0), dt_p)


# ======================================================================
#  A. WIRE IR-DROP  —  full nodal solve of the crossbar resistor network
# ======================================================================
class CrossbarIR:
    """Modified nodal analysis of an N x M 1R crossbar with wire resistance.

    Unknowns: row-wire node voltage Vr[i,j] and column-wire node voltage
    Vc[i,j] at every cross-point (2*N*M total). Each cross-point device is a
    conductance G[i,j] from Vr[i,j] to Vc[i,j]. Adjacent nodes along a wire are
    joined by a segment conductance gw = 1/r_wire. Rows are driven from their
    left end; columns are read (grounded through a final segment) at the bottom.

        I_col[j] = gw * Vc[N-1, j]          (current collected at column foot)

    The conductance matrix A depends only on G and r_wire, so we LU-factor it
    once and back-substitute for many input vectors (one VMM each).
    """

    def __init__(self, G, r_wire):
        self.N, self.M = G.shape
        N, M = self.N, self.M
        self.gw = 1.0 / r_wire
        gw = self.gw
        NM = N * M
        self.NM = NM

        # vectorised modified-nodal-analysis assembly (no per-cell Python loop,
        # so the matrix can be rebuilt+factored thousands of times in training)
        rgrid = np.arange(NM).reshape(N, M)               # row-wire node ids
        cgrid = (NM + np.arange(NM)).reshape(N, M)         # col-wire node ids

        # undirected edges as (a, b, conductance)
        ea = [rgrid.ravel()];  eb = [cgrid.ravel()];  eg = [G.ravel()]          # devices
        ea.append(rgrid[:, 1:].ravel());  eb.append(rgrid[:, :-1].ravel())      # row wires
        eg.append(np.full(N * (M - 1), gw))
        ea.append(cgrid[:-1, :].ravel()); eb.append(cgrid[1:, :].ravel())       # col wires
        eg.append(np.full((N - 1) * M, gw))
        A_, B_, Gc = np.concatenate(ea), np.concatenate(eb), np.concatenate(eg)

        diag = (np.bincount(A_, Gc, 2 * NM) + np.bincount(B_, Gc, 2 * NM))
        diag[rgrid[:, 0]] += gw           # driven sources at each row's left end
        diag[cgrid[-1, :]] += gw          # grounded foot of each column

        rows = np.concatenate([A_, B_, np.arange(2 * NM)])
        cols = np.concatenate([B_, A_, np.arange(2 * NM)])
        data = np.concatenate([-Gc, -Gc, diag])
        A = sp.csc_matrix((data, (rows, cols)), shape=(2 * NM, 2 * NM))
        self.lu = spla.splu(A)

    def _rhs(self, Vin):
        """Source vector(s): gw*Vin injected at each row's left-end node (j=0)."""
        Vin = np.atleast_2d(Vin)                 # (K, N)
        K = Vin.shape[0]
        b = np.zeros((2 * self.NM, K))
        idx = np.arange(self.N) * self.M         # j == 0 row nodes
        b[idx, :] = self.gw * Vin.T
        return b

    def vmm(self, Vin):
        """Bitline currents for one or many input vectors. Returns (K, M)."""
        b = self._rhs(Vin)
        x = self.lu.solve(b)                     # (2NM, K)
        Vc_bottom = x[self.NM + (self.N - 1) * self.M :
                      self.NM + self.N * self.M, :]      # (M, K)
        return (self.gw * Vc_bottom).T           # (K, M)

    def node_voltages(self, Vin):
        """Row-node voltage map for a single input (for visualisation)."""
        x = self.lu.solve(self._rhs(Vin))[:, 0]
        return x[:self.NM].reshape(self.N, self.M)


def figure_irdrop():
    rng = np.random.default_rng(3)

    # (a) voltage droop map: drive every row at VREAD into an all-LRS array
    N = 64
    G_on = np.full((N, N), 1.0 / 2.0e3)             # worst case: all low-R
    cb = CrossbarIR(G_on, r_wire=2.5)
    Vmap = cb.node_voltages(np.full(N, VREAD))

    # (b) output-current error vs array size, for two wire resistances
    sizes = [16, 32, 64, 128]
    rwires = [1.0, 2.5, 5.0]
    err = {rw: [] for rw in rwires}
    for n in sizes:
        G = np.full((n, n), 1.0 / 2.0e3) * np.exp(0.15 * rng.standard_normal((n, n)))
        Vin = np.full(n, VREAD)
        ideal = Vin @ G                              # Ohm + Kirchhoff, no wire R
        for rw in rwires:
            I = CrossbarIR(G, rw).vmm(Vin)[0]
            err[rw].append(100 * np.mean(np.abs(I - ideal) / ideal))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle("A. Wire IR-drop  —  full nodal solve of the crossbar network", fontsize=13)

    im = ax[0].imshow(Vmap * 1e3, cmap="viridis")
    ax[0].set(title="(a) Wordline voltage across a 64×64 all-LRS array\n"
                    "(driven at 200 mV from the left edge)",
              xlabel="column (→ far from driver)", ylabel="row")
    fig.colorbar(im, ax=ax[0], fraction=0.046, label="node voltage (mV)")

    for rw in rwires:
        ax[1].plot(sizes, err[rw], "o-", label=f"r_wire = {rw:.1f} Ω/cell")
    ax[1].set(title="(b) VMM error grows with array size (all-LRS worst case)",
              xlabel="array size N (N×N)", ylabel="mean |error| vs ideal (%)",
              xscale="log")
    ax[1].set_xticks(sizes); ax[1].set_xticklabels(sizes)
    ax[1].minorticks_off()                          # drop colliding 2×10¹ labels
    ax[1].grid(alpha=.3); ax[1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figA_irdrop.png", dpi=130)
    plt.close(fig)
    print("  wrote figA_irdrop.png")
    for n in sizes:
        line = "    N=%3d  " % n + "  ".join(
            f"r_w={rw}Ω:{err[rw][sizes.index(n)]:5.1f}%" for rw in rwires)
        print(line)


# ======================================================================
#  B. RETENTION / DRIFT  —  Arrhenius relaxation of the state variable
# ======================================================================
def drift_state(w, t_seconds, T_kelvin, Ea=1.0, tau0=4.8e-9, w_eq=0.02):
    """Relax w toward equilibrium: w(t) = w_eq + (w-w_eq)*exp(-t/tau(T))."""
    tau = tau0 * np.exp(Ea / (KB_EV * T_kelvin))
    return w_eq + (w - w_eq) * np.exp(-t_seconds / tau)


def figure_retention():
    # (a) normalised conductance vs time at several temperatures
    times = np.logspace(0, np.log10(3.15e7 * 3), 200)        # 1 s .. ~3 years
    temps_C = [25, 55, 85, 125]
    w0 = 0.95                                                # a programmed LRS cell
    g_on, g_off = 1.0 / 2.0e3, _GOFF_NOM
    def g_of_w(w): return w * g_on + (1 - w) * g_off
    g_full = g_of_w(w0)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    fig.suptitle("B. Retention / drift  —  the state relaxes thermally over time", fontsize=13)

    for Tc in temps_C:
        wt = drift_state(w0, times, Tc + 273.15)
        ax[0].semilogx(times, g_of_w(wt) / g_full, label=f"{Tc} °C")
    for mark, lab in [(3600, "1 h"), (86400, "1 d"), (2.6e6, "1 mo"), (3.15e7, "1 yr")]:
        ax[0].axvline(mark, color="grey", ls=":", lw=.7)
        ax[0].text(mark, 1.02, lab, rotation=90, fontsize=7, color="grey", va="bottom")
    ax[0].set(title="(a) LRS conductance retention (Ea=1.0 eV)",
              xlabel="time (s)", ylabel="conductance / initial", ylim=(0, 1.08))
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(title="temperature")

    # (b) a stored pattern fading: program "AlOx", then bake at 85 C
    from alox_crossbar import make_target_image
    img = make_target_image(64)
    g_lo, g_hi = 20e-6, 300e-6
    G_t = g_lo + img * (g_hi - g_lo)
    rng = np.random.default_rng(11)
    arr = variable_memristor((64, 64), rng)
    arr.w[...] = 0.02
    program_verify(arr, G_t)
    w_written = arr.w.copy()

    snaps = [(0, "t = 0"), (86400, "1 day"), (7 * 86400, "1 week"), (30 * 86400, "1 month")]
    panel = np.concatenate([
        np.clip((g_of_w(drift_state(w_written, t, 85 + 273.15)) - g_lo) / (g_hi - g_lo), 0, 1)
        for t, _ in snaps], axis=1)
    ax[1].imshow(panel, cmap="magma")
    ax[1].set_title("(b) Stored weights baked at 85 °C — memory fades")
    ax[1].set_xticks([32 + 64 * k for k in range(4)])
    ax[1].set_xticklabels([s for _, s in snaps]); ax[1].set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figB_retention.png", dpi=130)
    plt.close(fig)
    print("  wrote figB_retention.png")
    for Tc in temps_C:
        tau = 4.8e-9 * np.exp(1.0 / (KB_EV * (Tc + 273.15)))
        print(f"    retention time-constant @ {Tc:3d} °C : {tau:9.2e} s  ({tau/86400:8.2f} days)")


# ======================================================================
#  C. INFERENCE  —  conv building block + end-to-end classifier accuracy
# ======================================================================
def deploy_weights(W, g_lo=20e-6, g_hi=300e-6):
    """Map a signed weight matrix (rows x cols) to a differential G+/G- pair.

    Returns target conductances (rows x 2*cols, interleaved +,-) and the decode
    scale so that  (I+ - I-) / scale  recovers  Vin @ W.
    """
    half = (g_hi - g_lo) / 2.0
    g_mid = (g_hi + g_lo) / 2.0
    wmax = np.abs(W).max()
    Wn = W / wmax
    Gp = g_mid + Wn * half
    Gm = g_mid - Wn * half
    rows, cols = W.shape
    G_t = np.empty((rows, 2 * cols))
    G_t[:, 0::2] = Gp
    G_t[:, 1::2] = Gm
    decode_scale = 2 * half / wmax          # (I+ - I-) = decode_scale * (Vin @ W)
    return G_t, decode_scale


def read_after(write_mode, G_t, rng, drift_t=0.0, drift_T=358.15):
    """Return conductances actually held by a variable array after writing.

    write_mode: 'open' (set nominal w, variability uncorrected) or
                'verify' (closed-loop program-verify, variability corrected).
    drift_t > 0 relaxes the written state for drift_t seconds at drift_T.
    """
    arr = variable_memristor(G_t.shape, rng)
    if write_mode == "open":
        arr.w[...] = w_for_conductance(G_t)
    else:
        arr.w[...] = 0.02
        program_verify(arr, G_t)
    if drift_t > 0:
        arr.w = drift_state(arr.w, drift_t, drift_T)
    return arr.read_conductance(VREAD)


def figure_inference():
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(5)

    # ---- train a software linear classifier that FITS the array (64 -> 10) ----
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1) / 16.0     # (n, 64) in [0,1]
    y = digits.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=0)
    clf = LogisticRegression(max_iter=2000, C=5.0).fit(Xtr, ytr)
    W = clf.coef_.T                       # (64, 10)
    b = clf.intercept_                    # (10,)
    acc_sw = clf.score(Xte, yte)

    # deploy to a differential crossbar (64 rows x 20 cols)
    G_t, dscale = deploy_weights(W)
    Vin = Xte * VREAD                     # sub-threshold read voltages (Nte, 64)

    def accuracy(G, use_irdrop=False, r_wire=2.5):
        if use_irdrop:
            I = CrossbarIR(G, r_wire).vmm(Vin)            # (Nte, 20)
        else:
            I = Vin @ G                                   # ideal Ohm+Kirchhoff
        # (I+ - I-)/dscale = Vin@W = VREAD*(X@W); divide out VREAD, then add bias
        logits = (I[:, 0::2] - I[:, 1::2]) / (dscale * VREAD) + b
        return np.mean(logits.argmax(1) == yte)

    # scenarios
    G_ideal  = G_t                                              # perfect write
    G_open   = read_after("open",   G_t, np.random.default_rng(20))
    G_verify = read_after("verify", G_t, np.random.default_rng(21))
    scen = {
        "software\n(float)":      acc_sw,
        "crossbar\nideal write":  accuracy(G_ideal),
        "open-loop write\n(raw variability)": accuracy(G_open),
        "program-verify\nwrite":  accuracy(G_verify),
        "+ IR-drop\n(r=2.5Ω)":    accuracy(G_verify, use_irdrop=True, r_wire=2.5),
    }

    # accuracy vs retention time at 85 C (using program-verify write)
    ret_t = np.array([0, 3600, 86400, 7 * 86400, 30 * 86400, 180 * 86400, 365 * 86400.])
    ret_acc = []
    for t in ret_t:
        G_dr = read_after("verify", G_t, np.random.default_rng(22), drift_t=t)
        ret_acc.append(accuracy(G_dr))

    # ---- YOLO building block: one conv layer on the crossbar ----
    img = torch.tensor(digits.images[7] / 16.0, dtype=torch.float32)[None, None]  # 1x1x8x8
    Cout, k = 8, 3
    Wc = torch.randn(Cout, 1, k, k) * 0.5
    ref = F.conv2d(img, Wc, padding=1)[0].reshape(Cout, -1).T.numpy()             # (64patch, 8)
    patches = F.unfold(img, k, padding=1)[0].T.numpy()                            # (64, 9)
    Wc_mat = Wc.reshape(Cout, -1).T.numpy()                                       # (9, 8)
    Gc_t, cscale = deploy_weights(Wc_mat)
    Gc = read_after("verify", Gc_t, np.random.default_rng(23))
    Iconv = (patches * VREAD) @ Gc                                                # (64, 16)
    hw = (Iconv[:, 0::2] - Iconv[:, 1::2]) / (cscale * VREAD)                      # (64, 8)
    cos = np.sum(ref * hw) / (np.linalg.norm(ref) * np.linalg.norm(hw))

    # ---------------------------- figure ----------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("C. Inference on the array  —  accuracy loss from non-idealities", fontsize=13)

    names = list(scen); vals = [scen[n] * 100 for n in names]
    colors = ["#444", "#2a7fb8", "#c1121f", "#2a9d8f", "#d9821a"]
    bars = ax[0].bar(range(len(names)), vals, color=colors)
    ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, fontsize=8)
    ax[0].set(title="(a) Digit-classifier accuracy (64→10 layer)",
              ylabel="test accuracy (%)", ylim=(0, 100))
    for bar, v in zip(bars, vals):
        ax[0].text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}",
                   ha="center", fontsize=8)
    ax[0].grid(alpha=.3, axis="y")

    ax[1].semilogx(np.maximum(ret_t, 1), np.array(ret_acc) * 100, "o-", color="#d9821a")
    ax[1].axhline(acc_sw * 100, ls="--", color="grey", lw=.8, label="software")
    ax[1].axhline(10, ls=":", color="k", lw=.8, label="chance (10 classes)")
    for mark, lab in [(3600, "1h"), (86400, "1d"), (2.6e6, "1mo"), (3.15e7, "1yr")]:
        ax[1].axvline(mark, color="grey", ls=":", lw=.5)
    ax[1].set(title="(b) Accuracy vs retention time @ 85 °C",
              xlabel="time since programming (s)", ylabel="accuracy (%)", ylim=(0, 100))
    ax[1].grid(alpha=.3, which="both"); ax[1].legend(fontsize=8)

    ax[2].scatter(ref.ravel(), hw.ravel(), s=8, alpha=.4, color="#2a9d8f")
    lim = [ref.min() * 1.1, ref.max() * 1.1]
    ax[2].plot(lim, lim, "k--", lw=.8)
    ax[2].set(title=f"(c) Conv layer (YOLO building block)\ncrossbar vs float — cosine sim = {cos:.4f}",
              xlabel="float conv output", ylabel="crossbar conv output")
    ax[2].grid(alpha=.3)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig("figC_inference.png", dpi=130)
    plt.close(fig)
    print("  wrote figC_inference.png")
    for n in names:
        print(f"    {n.replace(chr(10), ' '):38s}: {scen[n]*100:5.1f} %")
    print(f"    conv-layer cosine similarity (crossbar vs float): {cos:.4f}")
    print(f"    accuracy after 1 month @85°C: {ret_acc[4]*100:.1f} %  "
          f"(1 year: {ret_acc[-1]*100:.1f} %)")


# ======================================================================
if __name__ == "__main__":
    print("Crossbar non-idealities: IR-drop, retention, inference")
    print("-" * 56)
    print("[A] wire IR-drop nodal solve")
    figure_irdrop()
    print("[B] retention / thermal drift")
    figure_retention()
    print("[C] inference accuracy (conv block + digit classifier)")
    figure_inference()
    print("-" * 56)
    print("done. figA_irdrop.png, figB_retention.png, figC_inference.png")
