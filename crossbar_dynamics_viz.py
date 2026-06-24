"""
Two dynamic views of the trained crossbar
==========================================

  1. crossbar_inference_irdrop.gif
     Test-time inference run THROUGH the IR-drop network, using tiles that were
     trained with IR-drop in the loop. You see the decision boundary the array
     learned *around* the wire resistance still segment the spiral.

  2. crossbar_retention_drift.gif
     The same trained tiles, left to sit. Conductances relax thermally
     (Arrhenius drift at 85 °C); the stored weights fade toward zero and the
     decision boundary visibly dissolves until the spiral is unclassifiable.

Run:  python3 crossbar_dynamics_viz.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter

from crossbar_learn import (make_spirals, train_crossbar, train_crossbar_ir,
                            forward, softmax, layer_Gmat, accuracy)
from crossbar_advanced import CrossbarIR, drift_state

OR, BL = "#d9821a", "#2a7fb8"


def spiral_data():
    rng = np.random.default_rng(0)
    X, y = make_spirals(n_per=500, turns=1.5, noise=0.08, rng=rng)
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    Y = np.eye(2)[y]
    return X, y, Xa, Y


def boundary_grid(X):
    gx = np.linspace(X[:, 0].min() - .25, X[:, 0].max() + .25, 200)
    gy = np.linspace(X[:, 1].min() - .25, X[:, 1].max() + .25, 200)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel(), np.ones(GX.size)], 1)
    return gx, gy, GX, GY, grid


# ======================================================================
#  1. INFERENCE THROUGH THE IR-DROP NETWORK
# ======================================================================
def animate_irdrop():
    X, y, Xa, Y = spiral_data()
    r_wire = 1.0
    print("  [IR] training with IR-drop in the loop ...")
    acc, (t1, t2) = train_crossbar_ir(Xa, Y, y, 64, 10.0, 350,
                                      np.random.default_rng(5), 1.5e-3, 3e-3, 32, r_wire)
    W1, W2 = t1.weight(), t2.weight()
    print(f"  [IR] trained-in-loop accuracy {np.mean(acc[-20:])*100:.1f}%")

    # build each layer's resistive-network solver ONCE (weights fixed at test)
    cb1 = CrossbarIR(layer_Gmat(t1), r_wire)
    cb2 = CrossbarIR(layer_Gmat(t2), r_wire)

    def fwd(Xin):
        I1 = cb1.vmm(Xin)
        a1 = np.tanh((I1[:, 0::2] - I1[:, 1::2]) * t1.w_scale)
        a1a = np.concatenate([a1, np.ones((len(a1), 1))], 1)
        I2 = cb2.vmm(a1a)
        p = softmax((I2[:, 0::2] - I2[:, 1::2]) * t2.w_scale)
        return a1, p

    gx, gy, GX, GY, grid = boundary_grid(X)
    _, pg = fwd(grid)
    PB = pg[:, 1].reshape(GX.shape)

    # test spiral revealed centre-outward
    Xt, yt = make_spirals(n_per=80, turns=1.5, noise=0.06, rng=np.random.default_rng(99))
    order = np.argsort(np.hypot(Xt[:, 0], Xt[:, 1]))
    Xt, yt = Xt[order], yt[order]
    Xta = np.concatenate([Xt, np.ones((len(Xt), 1))], 1)

    fig = plt.figure(figsize=(13, 6))
    fig.suptitle("Inference THROUGH the IR-drop network (r = 1 Ω/cell) — "
                 "boundary learned around wire resistance", fontsize=12)
    gs = GridSpec(2, 2, width_ratios=[1, 1.7], hspace=.4, wspace=.28,
                  left=.07, right=.98, top=.9, bottom=.1)
    axH = fig.add_subplot(gs[0, 0]); axO = fig.add_subplot(gs[1, 0])
    axS = fig.add_subplot(gs[:, 1])

    him = axH.imshow(np.zeros((8, 8)), cmap="RdBu_r", vmin=-1, vmax=1)
    axH.set_title("Hidden neurons firing (via IR-drop VMM)", fontsize=9)
    axH.set_xticks([]); axH.set_yticks([])
    obars = axO.bar([0, 1], [0, 0], color=[OR, BL])
    axO.set_ylim(0, 1); axO.set_xticks([0, 1]); axO.set_xticklabels(["class A", "class B"], fontsize=9)
    axO.set_title("Class currents → decision", fontsize=9)
    otxt = axO.text(.5, .9, "", transform=axO.transAxes, ha="center", fontsize=11, weight="bold")

    axS.contourf(GX, GY, PB, levels=24, cmap="RdBu", alpha=.45)
    axS.scatter(*X[y == 0].T, s=4, c=OR, alpha=.12)
    axS.scatter(*X[y == 1].T, s=4, c=BL, alpha=.12)
    axS.set_title("Segmenting the spiral on the wire-resistive array", fontsize=10)
    axS.set_xlabel("x₁"); axS.set_ylabel("x₂"); axS.set_aspect("equal")
    axS.set_xlim(gx.min(), gx.max()); axS.set_ylim(gy.min(), gy.max())
    acc_txt = axS.text(.02, .97, "", transform=axS.transAxes, va="top", fontsize=10,
                       bbox=dict(boxstyle="round", fc="white", alpha=.8))
    star = axS.plot([], [], marker="*", ms=22, mfc="yellow", mec="k", mew=1.2, zorder=5)[0]
    n_ok = [0]

    def update(i):
        a1, p = fwd(Xta[i][None]); a1, p = a1[0], p[0]
        pred, true = int(p.argmax()), int(yt[i]); ok = pred == true
        n_ok[0] += ok
        him.set_data(a1.reshape(8, 8))
        for k, b in enumerate(obars):
            b.set_height(p[k])
        otxt.set_text(f"→ class {'A' if pred == 0 else 'B'}  ({p[pred]*100:.0f}%)")
        otxt.set_color(OR if pred == 0 else BL)
        x0, x1 = Xta[i, 0], Xta[i, 1]
        if ok:
            axS.scatter([x0], [x1], s=42, c=(OR if pred == 0 else BL),
                        edgecolors="k", linewidths=.4, zorder=4)
        else:
            axS.scatter([x0], [x1], s=70, c="k", marker="x", linewidths=1.8, zorder=4)
        star.set_data([x0], [x1])
        acc_txt.set_text(f"inferred {i+1}/{len(Xta)}\nlive accuracy {100*n_ok[0]/(i+1):.0f}%")
        return him, star

    print("  [IR] rendering ...")
    anim = FuncAnimation(fig, update, frames=len(Xta), interval=90, blit=False)
    anim.save("crossbar_inference_irdrop.gif", writer=PillowWriter(fps=11), dpi=90)
    star.set_data([], [])
    fig.savefig("crossbar_inference_irdrop_final.png", dpi=110)
    plt.close(fig)
    print("  [IR] wrote crossbar_inference_irdrop.gif")


# ======================================================================
#  2. RETENTION DRIFT — the boundary dissolves over time
# ======================================================================
def animate_drift():
    X, y, Xa, Y = spiral_data()
    print("  [drift] training crossbar (variable devices) ...")
    acc, (t1, t2) = train_crossbar(Xa, Y, y, 64, 10.0, 400,
                                   np.random.default_rng(5), False, 1.5e-3, 3e-3, 32)
    print(f"  [drift] trained accuracy {acc[-1]*100:.1f}%")

    # held-out test spiral for the live accuracy number
    Xt, yt = make_spirals(n_per=80, turns=1.5, noise=0.06, rng=np.random.default_rng(99))
    Xtta = np.concatenate([Xt, np.ones((len(Xt), 1))], 1)

    # snapshot the just-trained state; we relax copies of it
    w0 = [(t1.Gp.w.copy(), t1.Gm.w.copy()), (t2.Gp.w.copy(), t2.Gm.w.copy())]
    W10, W20 = t1.weight(), t2.weight()
    wlim1, wlim2 = np.abs(W10).max(), np.abs(W20).max()

    gx, gy, GX, GY, grid = boundary_grid(X)
    T_K = 85 + 273.15
    times = np.concatenate([[0.0], np.logspace(2, np.log10(40 * 86400), 59)])  # 60 frames

    def set_time(t):
        t1.Gp.w = drift_state(w0[0][0], t, T_K); t1.Gm.w = drift_state(w0[0][1], t, T_K)
        t2.Gp.w = drift_state(w0[1][0], t, T_K); t2.Gm.w = drift_state(w0[1][1], t, T_K)
        return t1.weight(), t2.weight()

    fig = plt.figure(figsize=(13, 6))
    fig.suptitle("Retention drift @ 85 °C — the stored weights fade and the "
                 "decision boundary dissolves", fontsize=12)
    gs = GridSpec(2, 2, width_ratios=[1, 1.7], hspace=.4, wspace=.28,
                  left=.07, right=.98, top=.9, bottom=.1)
    axW1 = fig.add_subplot(gs[0, 0]); axW2 = fig.add_subplot(gs[1, 0])
    axS = fig.add_subplot(gs[:, 1])

    imW1 = axW1.imshow(W10, cmap="PuOr", vmin=-wlim1, vmax=wlim1, aspect="auto")
    axW1.set_title("Tile 1 weights (input→hidden)", fontsize=9)
    axW1.set_yticks([0, 1, 2]); axW1.set_yticklabels(["x₁", "x₂", "bias"], fontsize=8)
    axW1.set_xlabel("hidden unit", fontsize=8)
    imW2 = axW2.imshow(W20.T, cmap="PuOr", vmin=-wlim2, vmax=wlim2, aspect="auto")
    axW2.set_title("Tile 2 weights (hidden→class)", fontsize=9)
    axW2.set_yticks([0, 1]); axW2.set_yticklabels(["A", "B"], fontsize=8)
    axW2.set_xlabel("hidden unit", fontsize=8)

    def human(t):
        if t < 90: return f"{t:.0f} s"
        if t < 5400: return f"{t/60:.0f} min"
        if t < 1.8 * 86400: return f"{t/3600:.1f} h"
        return f"{t/86400:.1f} days"

    def update(i):
        t = times[i]
        W1, W2 = set_time(t)
        _, _, _, pg = forward(grid, W1, W2)
        PB = pg[:, 1].reshape(GX.shape)
        _, _, _, pt = forward(Xtta, W1, W2)
        acc_now = np.mean(pt.argmax(1) == yt)

        imW1.set_data(W1); imW2.set_data(W2.T)
        axS.clear()
        axS.contourf(GX, GY, PB, levels=24, cmap="RdBu", alpha=.55, vmin=0, vmax=1)
        axS.scatter(*X[y == 0].T, s=5, c=OR, alpha=.30)
        axS.scatter(*X[y == 1].T, s=5, c=BL, alpha=.30)
        axS.set_xlim(gx.min(), gx.max()); axS.set_ylim(gy.min(), gy.max())
        axS.set_aspect("equal"); axS.set_xlabel("x₁"); axS.set_ylabel("x₂")
        axS.set_title("Decision boundary vs. time since programming", fontsize=10)
        axS.text(.02, .97, f"t = {human(t)}  @ 85 °C\naccuracy {acc_now*100:.0f}%",
                 transform=axS.transAxes, va="top", fontsize=11,
                 bbox=dict(boxstyle="round", fc="white", alpha=.85))
        return imW1, imW2

    print("  [drift] rendering ...")
    anim = FuncAnimation(fig, update, frames=len(times), interval=140, blit=False)
    anim.save("crossbar_retention_drift.gif", writer=PillowWriter(fps=8), dpi=90)
    fig.savefig("crossbar_retention_drift_final.png", dpi=110)
    plt.close(fig)
    # restore the trained state
    t1.Gp.w, t1.Gm.w = w0[0]; t2.Gp.w, t2.Gm.w = w0[1]
    print("  [drift] wrote crossbar_retention_drift.gif")


if __name__ == "__main__":
    print("Crossbar dynamic visualisations")
    print("-" * 40)
    animate_irdrop()
    animate_drift()
    print("-" * 40)
    print("done. crossbar_inference_irdrop.gif, crossbar_retention_drift.gif")
