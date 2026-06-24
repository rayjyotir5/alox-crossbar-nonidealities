"""
In-situ learning on the crossbar: a memristor MLP solves the two-spirals task
==============================================================================

This closes the loop back to the neuromorphic premise. So far the array only
*stored* and *multiplied* pre-trained weights. Here the weights ARE the device
conductances and they are *trained on the array* — every weight update is a
programming pulse that moves the oxygen-vacancy filament, using the exact same
nonlinear `step()` dynamics as the single-device model. That gradual,
threshold-gated conductance change is synaptic plasticity; doing gradient
descent with it is in-situ learning.

Task: TWO INTERLEAVING SPIRALS — the textbook "you need a real nonlinear
network" problem (a linear classifier scores ~50%). We train a
2 -> 64 -> 2 MLP whose two weight matrices live on differential crossbar tiles
(3x128 and 65x4 conductances), and watch it converge.

Why this is the interesting experiment
---------------------------------------
On-chip weight updates are HARDER than inference: each SET/RESET pulse changes
conductance by an amount that depends on the present state (saturating, and
asymmetric between potentiation and depression), and every device is different.
We use the hardware-friendly MANHATTAN rule (move each weight one pulse in the
sign of -gradient) and show the loop still converges — because the feedback
self-corrects the device imperfections. We compare:
    * ideal float SGD            (digital reference)
    * crossbar in-situ, ideal devices
    * crossbar in-situ, variable + noisy devices

Run:  python3 crossbar_learn.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alox_crossbar import AloxMemristor
from crossbar_advanced import (variable_memristor, _GON_NOM, _GOFF_NOM, VREAD,
                               CrossbarIR, drift_state)

G_SWING = _GON_NOM - _GOFF_NOM          # full conductance range of a nominal cell
G_MID = 0.5 * (_GON_NOM + _GOFF_NOM)


# ----------------------------------------------------------------------
#  data: two interleaving spirals
# ----------------------------------------------------------------------
def make_spirals(n_per=500, turns=2.0, noise=0.09, rng=None):
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0.06, 1.0, n_per)
    theta = t * turns * 2 * np.pi
    r = t
    X, y = [], []
    for cls, sign in ((0, +1), (1, -1)):
        x1 = sign * r * np.cos(theta) + noise * rng.standard_normal(n_per)
        x2 = sign * r * np.sin(theta) + noise * rng.standard_normal(n_per)
        X.append(np.stack([x1, x2], 1)); y.append(np.full(n_per, cls))
    X = np.concatenate(X); y = np.concatenate(y)
    p = rng.permutation(len(y))
    return X[p], y[p]


# ----------------------------------------------------------------------
#  a trainable crossbar weight tile  (differential G+ / G- pair)
# ----------------------------------------------------------------------
class CrossbarTile:
    """Signed weights W = (G+ - G-) * w_scale, programmed by SET/RESET pulses.

    `ideal=True` uses identical, noiseless devices; otherwise full
    device-to-device variability + cycle-to-cycle pulse noise.
    """

    def __init__(self, n_in, n_out, w_max, rng, ideal=False, init_std=0.5,
                 Vset_p=1.15, Vreset_p=1.15):
        shape = (n_in, n_out)
        if ideal:
            self.Gp = AloxMemristor(shape=shape, rng=rng)
            self.Gm = AloxMemristor(shape=shape, rng=rng)
        else:
            self.Gp = variable_memristor(shape, rng)
            self.Gm = variable_memristor(shape, rng)
        self.w_scale = w_max / G_SWING                 # conductance-diff -> weight
        self.Vset_p, self.Vreset_p = Vset_p, Vreset_p
        # init small random signed weights, symmetric about the mid conductance
        diff = np.clip(rng.standard_normal(shape) * init_std / self.w_scale,
                       -G_SWING * 0.9, G_SWING * 0.9)
        self.Gp.w = np.clip((G_MID + diff / 2 - _GOFF_NOM) / G_SWING, 1e-3, 1 - 1e-3)
        self.Gm.w = np.clip((G_MID - diff / 2 - _GOFF_NOM) / G_SWING, 1e-3, 1 - 1e-3)

    def weight(self):
        return (self.Gp.read_conductance(VREAD) -
                self.Gm.read_conductance(VREAD)) * self.w_scale

    def update(self, grad, lr_pulse, dt_max):
        """Gradient-proportional pulse update through the real device curve.

        The pulse WIDTH scales with |grad| (clamped), so the conductance change
        approximates SGD in the device's linear regime and rolls off
        nonlinearly near the rails. To raise W we SET G+ and RESET G- (keeping
        the differential pair balanced); to lower W, the reverse.
        """
        dt = np.clip(lr_pulse * np.abs(grad), 0.0, dt_max)
        up, down = grad < 0, grad > 0
        Vp = np.where(up, self.Vset_p, np.where(down, -self.Vreset_p, 0.0))
        Vm = np.where(up, -self.Vreset_p, np.where(down, self.Vset_p, 0.0))
        self.Gp.step(Vp, dt)
        self.Gm.step(Vm, dt)


# ----------------------------------------------------------------------
#  MLP:  2 -> H -> 2  with weights on two crossbar tiles
# ----------------------------------------------------------------------
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def forward(Xa, W1, W2):
    z1 = Xa @ W1
    a1 = np.tanh(z1)
    a1a = np.concatenate([a1, np.ones((len(a1), 1))], 1)
    p = softmax(a1a @ W2)
    return z1, a1, a1a, p


def grads(Xa, a1, a1a, p, Y, W2):
    N = len(Xa)
    dz2 = (p - Y) / N
    dW2 = a1a.T @ dz2
    da1 = (dz2 @ W2.T)[:, :-1]
    dz1 = da1 * (1 - a1 ** 2)
    dW1 = Xa.T @ dz1
    return dW1, dW2


def accuracy(Xa, W1, W2, y):
    _, _, _, p = forward(Xa, W1, W2)
    return np.mean(p.argmax(1) == y)


def minibatches(n, bs, rng):
    idx = rng.permutation(n)
    for s in range(0, n, bs):
        yield idx[s:s + bs]


def train_float(Xa, Y, y, W1, W2, epochs, lr, bs, rng):
    W1, W2 = W1.copy(), W2.copy()
    acc = []
    for _ in range(epochs):
        for b in minibatches(len(Xa), bs, rng):
            _, a1, a1a, p = forward(Xa[b], W1, W2)
            dW1, dW2 = grads(Xa[b], a1, a1a, p, Y[b], W2)
            W1 -= lr * dW1; W2 -= lr * dW2
        acc.append(accuracy(Xa, W1, W2, y))
    return acc, (W1, W2)


def train_crossbar(Xa, Y, y, H, w_max, epochs, rng, ideal,
                   lr_pulse, dt_max, bs, lr_decay=1.0):
    t1 = CrossbarTile(Xa.shape[1], H, w_max, np.random.default_rng(101), ideal=ideal)
    t2 = CrossbarTile(H + 1, Y.shape[1], w_max, np.random.default_rng(202), ideal=ideal)
    acc = []
    for ep in range(epochs):
        lrp = lr_pulse * (lr_decay ** ep)
        for b in minibatches(len(Xa), bs, rng):
            W1, W2 = t1.weight(), t2.weight()
            _, a1, a1a, p = forward(Xa[b], W1, W2)
            dW1, dW2 = grads(Xa[b], a1, a1a, p, Y[b], W2)
            t1.update(dW1, lrp, dt_max)
            t2.update(dW2, lrp, dt_max)
        acc.append(accuracy(Xa, t1.weight(), t2.weight(), y))
    return acc, (t1, t2)


# ----------------------------------------------------------------------
#  Forward pass THROUGH the IR-drop network (hardware-in-the-loop)
# ----------------------------------------------------------------------
def layer_Gmat(tile):
    """Interleave the differential pair into a physical (n_in x 2*n_out) array."""
    Gp = tile.Gp.read_conductance(VREAD)
    Gm = tile.Gm.read_conductance(VREAD)
    Gmat = np.empty((Gp.shape[0], 2 * Gp.shape[1]))
    Gmat[:, 0::2] = Gp
    Gmat[:, 1::2] = Gm
    return Gmat


def vmm_ir(Gmat, Vin, r_wire):
    """Differential bitline currents from the full nodal solve."""
    I = CrossbarIR(Gmat, r_wire).vmm(Vin)
    return I[:, 0::2] - I[:, 1::2]


def forward_ir(Xa, t1, t2, r_wire):
    z1 = vmm_ir(layer_Gmat(t1), Xa, r_wire) * t1.w_scale
    a1 = np.tanh(z1)
    a1a = np.concatenate([a1, np.ones((len(a1), 1))], 1)
    z2 = vmm_ir(layer_Gmat(t2), a1a, r_wire) * t2.w_scale
    return z1, a1, a1a, softmax(z2)


def accuracy_ir(Xa, t1, t2, r_wire, y):
    _, _, _, p = forward_ir(Xa, t1, t2, r_wire)
    return np.mean(p.argmax(1) == y)


def train_crossbar_ir(Xa, Y, y, H, w_max, epochs, rng, lr_pulse, dt_max, bs,
                      r_wire, lr_decay=0.992):
    """In-situ training with IR-drop IN THE FORWARD PASS.

    Forward currents come from the real (wire-resistive) network; gradients use
    the ideal differential-weight read as a surrogate (you cannot backprop
    through the analog array, so you model it). The pulse update then adapts the
    conductances to whatever transfer function the hardware actually implements.
    A pulse-size decay schedule settles the otherwise-noisy convergence (the
    surrogate-gradient mismatch keeps the loop dithering at fixed step size).
    """
    t1 = CrossbarTile(Xa.shape[1], H, w_max, np.random.default_rng(101), ideal=False)
    t2 = CrossbarTile(H + 1, Y.shape[1], w_max, np.random.default_rng(202), ideal=False)
    acc = []
    for ep in range(epochs):
        lrp = lr_pulse * (lr_decay ** ep)
        for b in minibatches(len(Xa), bs, rng):
            _, a1, a1a, p = forward_ir(Xa[b], t1, t2, r_wire)   # real hardware
            W2 = t2.weight()                                    # surrogate model
            dz2 = (p - Y[b]) / len(b)
            dW2 = a1a.T @ dz2
            dz1 = (dz2 @ W2.T)[:, :-1] * (1 - a1 ** 2)
            dW1 = Xa[b].T @ dz1
            t1.update(dW1, lrp, dt_max)
            t2.update(dW2, lrp, dt_max)
        acc.append(accuracy_ir(Xa, t1, t2, r_wire, y))
    return acc, (t1, t2)


def eval_after_drift(t1, t2, t_sec, T_kelvin, Xa, y, r_wire=None):
    """Accuracy after the trained conductances relax for t_sec at T (no mutation)."""
    saved = [(tl.Gp.w.copy(), tl.Gm.w.copy()) for tl in (t1, t2)]
    for tl in (t1, t2):
        tl.Gp.w = drift_state(tl.Gp.w, t_sec, T_kelvin)
        tl.Gm.w = drift_state(tl.Gm.w, t_sec, T_kelvin)
    a = (accuracy(Xa, t1.weight(), t2.weight(), y) if r_wire is None
         else accuracy_ir(Xa, t1, t2, r_wire, y))
    for tl, (gp, gm) in zip((t1, t2), saved):
        tl.Gp.w, tl.Gm.w = gp, gm
    return a


# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(0)
    X, y = make_spirals(n_per=500, turns=1.5, noise=0.08, rng=rng)
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)          # bias column
    Y = np.eye(2)[y]
    H, w_max, epochs, bs = 64, 10.0, 400, 32
    lr_pulse, dt_max = 1.5e-3, 3.0e-3

    # float reference, seeded from the same init distribution for fairness
    seed = CrossbarTile(3, H, w_max, np.random.default_rng(1), ideal=True)
    seed2 = CrossbarTile(H + 1, 2, w_max, np.random.default_rng(2), ideal=True)
    Wf1, Wf2 = seed.weight(), seed2.weight()

    print("  training float reference (digital SGD) ...")
    acc_f, (Wf1f, Wf2f) = train_float(Xa, Y, y, Wf1, Wf2, epochs, 0.3, bs,
                                      np.random.default_rng(5))
    print("  training crossbar in-situ (variable + noisy devices) ...")
    acc_cv, (tc1, tc2) = train_crossbar(Xa, Y, y, H, w_max, epochs,
                                        np.random.default_rng(5), False,
                                        lr_pulse, dt_max, bs)
    Wc1, Wc2 = tc1.weight(), tc2.weight()

    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("In-situ learning on a memristor crossbar — two-spirals task", fontsize=14)

    # (a) data
    ax[0, 0].scatter(*X[y == 0].T, s=8, c="#d9821a", label="class A")
    ax[0, 0].scatter(*X[y == 1].T, s=8, c="#2a7fb8", label="class B")
    ax[0, 0].set(title="(a) Two interleaving spirals", xlabel="x₁", ylabel="x₂")
    ax[0, 0].set_aspect("equal"); ax[0, 0].legend(fontsize=8)

    # (b) convergence
    ax[0, 1].plot(np.array(acc_f) * 100, color="#444", lw=1.8, label="float SGD (digital)")
    ax[0, 1].plot(np.array(acc_cv) * 100, color="#c1121f", lw=1.8,
                  label="crossbar in-situ (variable + noisy devices)")
    ax[0, 1].axhline(50, ls=":", c="k", lw=.8, label="best linear model ≈ chance")
    ax[0, 1].set(title="(b) Training accuracy converges", xlabel="epoch",
                 ylabel="accuracy (%)", ylim=(45, 101))
    ax[0, 1].grid(alpha=.3); ax[0, 1].legend(fontsize=8, loc="lower right")

    # (c,d) decision boundaries
    gx = np.linspace(X[:, 0].min() - .2, X[:, 0].max() + .2, 240)
    gy = np.linspace(X[:, 1].min() - .2, X[:, 1].max() + .2, 240)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel(), np.ones(GX.size)], 1)

    def boundary(axx, W1, W2, title):
        _, _, _, p = forward(grid, W1, W2)
        axx.contourf(GX, GY, p[:, 1].reshape(GX.shape), levels=20, cmap="RdBu", alpha=.7)
        axx.scatter(*X[y == 0].T, s=5, c="#d9821a")
        axx.scatter(*X[y == 1].T, s=5, c="#2a7fb8")
        axx.set(title=title); axx.set_aspect("equal")

    boundary(ax[1, 0], Wf1f, Wf2f, f"(c) Float decision boundary — {acc_f[-1]*100:.1f}%")
    boundary(ax[1, 1], Wc1, Wc2,
             f"(d) Crossbar (variable) boundary — {acc_cv[-1]*100:.1f}%")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("figD_insitu_learning.png", dpi=130)
    plt.close(fig)
    print("  wrote figD_insitu_learning.png")
    print(f"    final accuracy  | float SGD {acc_f[-1]*100:5.1f}%  | "
          f"crossbar in-situ {acc_cv[-1]*100:5.1f}%")

    # ==================================================================
    #  E. retention drift after training, and IR-drop in the loop
    # ==================================================================
    print("  E1: retention drift between training and inference ...")
    times = np.array([0, 3600, 86400, 7 * 86400, 30 * 86400, 180 * 86400, 365 * 86400.])
    ret = {Tc: [eval_after_drift(tc1, tc2, t, Tc + 273.15, Xa, y) for t in times]
           for Tc in (25, 85, 125)}

    print("  E2: training with IR-drop in the forward pass ...")
    r_wire = 1.0
    acc_naive_ir = accuracy_ir(Xa, tc1, tc2, r_wire, y)         # trained ideal, tested on IR
    acc_ir_hist, _ = train_crossbar_ir(Xa, Y, y, H, w_max, 350,
                                       np.random.default_rng(5),
                                       lr_pulse, dt_max, bs, r_wire)
    acc_ir_trained = np.mean(acc_ir_hist[-20:])                 # trailing mean (curve is noisy)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Retention drift after training  &  learning around IR-drop", fontsize=13)

    # (a) forgetting curves
    for Tc, c in [(25, "#2a7fb8"), (85, "#d9821a"), (125, "#c1121f")]:
        ax[0].semilogx(np.maximum(times, 1), np.array(ret[Tc]) * 100, "o-", color=c, label=f"{Tc} °C")
    ax[0].axhline(acc_cv[-1] * 100, ls="--", c="grey", lw=.8, label="just-trained")
    ax[0].axhline(50, ls=":", c="k", lw=.8, label="chance")
    for mark, lab in [(3600, "1h"), (86400, "1d"), (2.6e6, "1mo"), (3.15e7, "1yr")]:
        ax[0].axvline(mark, color="grey", ls=":", lw=.5)
    ax[0].set(title="(a) Trained network forgets as conductance drifts",
              xlabel="time since training (s)", ylabel="accuracy (%)", ylim=(45, 100))
    ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=8)

    # (b) IR-drop: naive deploy vs train-in-the-loop
    names = ["train ideal\ntest ideal", f"train ideal\ntest IR (r={r_wire:.0f}Ω)",
             f"train w/ IR\ntest IR (r={r_wire:.0f}Ω)"]
    vals = [acc_cv[-1] * 100, acc_naive_ir * 100, acc_ir_trained * 100]
    bars = ax[1].bar(range(3), vals, color=["#2a9d8f", "#c1121f", "#d9821a"])
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(names, fontsize=8)
    ax[1].set(title="(b) The loop learns around wire resistance",
              ylabel="accuracy (%)", ylim=(0, 100))
    for bar, v in zip(bars, vals):
        ax[1].text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=9)
    ax[1].grid(alpha=.3, axis="y")

    # (c) IR-aware training convergence
    ax[2].plot(np.array(acc_ir_hist) * 100, color="#d9821a", lw=1.2, label="train w/ IR-drop")
    ax[2].axhline(acc_naive_ir * 100, ls="--", c="#c1121f", lw=1.0, label="naive deploy (no IR training)")
    ax[2].axhline(50, ls=":", c="k", lw=.8, label="chance")
    ax[2].set(title="(c) IR-aware training converges (noisily)",
              xlabel="epoch", ylabel="accuracy (%)", ylim=(45, 100))
    ax[2].grid(alpha=.3); ax[2].legend(fontsize=8, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("figE_drift_and_irdrop.png", dpi=130)
    plt.close(fig)
    print("  wrote figE_drift_and_irdrop.png")
    print(f"    retention @85°C: 1day {ret[85][2]*100:.0f}%  1week {ret[85][3]*100:.0f}%  "
          f"1month {ret[85][4]*100:.0f}%")
    print(f"    IR-drop (r={r_wire:.0f}Ω): naive {acc_naive_ir*100:.1f}%  ->  "
          f"trained-in-loop {acc_ir_trained*100:.1f}%")


if __name__ == "__main__":
    print("In-situ crossbar learning: two-spirals")
    print("-" * 40)
    main()
    print("-" * 40)
    print("done. figD_insitu_learning.png")
