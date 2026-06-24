"""
Test-time inference on the trained crossbar — watch it segment the spiral
=========================================================================

Trains the in-situ spiral MLP (from crossbar_learn) on the variable/noisy
crossbar, then ANIMATES test-time inference:

  * left  : the trained weights as they are physically encoded — two crossbar
            conductance tiles (input->hidden and hidden->class). Static; this
            is what is "stored in the array".
  * middle: the live forward pass for the current test point — the 64 hidden
            neurons firing (tanh activations) and the two output class
            currents. Changes every frame.
  * right : the spiral plane. Each test point is pushed through the array,
            classified, and dropped onto the map coloured by the array's
            prediction (✗ if wrong). The learned decision boundary is the
            background, so you watch the two spiral arms get carved out.

Run:  python3 crossbar_inference_viz.py
Outputs crossbar_inference.gif (+ a final-frame PNG).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter

from crossbar_learn import (make_spirals, train_crossbar, forward, softmax)

OR, BL = "#d9821a", "#2a7fb8"          # class A (orange), class B (blue)


def main():
    rng = np.random.default_rng(0)
    # ---- train the crossbar in-situ (variable + noisy devices) ----
    X, y = make_spirals(n_per=500, turns=1.5, noise=0.08, rng=rng)
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    Y = np.eye(2)[y]
    H, w_max = 64, 10.0
    print("  training crossbar (variable devices) ...")
    acc, (t1, t2) = train_crossbar(Xa, Y, y, H, w_max, 400,
                                   np.random.default_rng(5), False, 1.5e-3, 3e-3, 32)
    W1, W2 = t1.weight(), t2.weight()
    print(f"  trained accuracy {acc[-1]*100:.1f}%")

    # ---- a fresh test spiral, revealed centre-outward (sorted by radius) ----
    Xt, yt = make_spirals(n_per=80, turns=1.5, noise=0.06, rng=np.random.default_rng(99))
    order = np.argsort(np.hypot(Xt[:, 0], Xt[:, 1]))
    Xt, yt = Xt[order], yt[order]
    Xta = np.concatenate([Xt, np.ones((len(Xt), 1))], 1)

    # ---- precompute the decision-boundary background ----
    gx = np.linspace(X[:, 0].min() - .25, X[:, 0].max() + .25, 220)
    gy = np.linspace(X[:, 1].min() - .25, X[:, 1].max() + .25, 220)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel(), np.ones(GX.size)], 1)
    _, _, _, pg = forward(grid, W1, W2)
    PB = pg[:, 1].reshape(GX.shape)

    # ============================ figure layout ============================
    fig = plt.figure(figsize=(15, 6.2))
    fig.suptitle("Test-time inference on the trained memristor crossbar", fontsize=14)
    gs = GridSpec(2, 3, width_ratios=[1.05, 1.05, 1.7], height_ratios=[1, 1],
                  hspace=0.42, wspace=0.32, left=.05, right=.985, top=.9, bottom=.09)

    axW1 = fig.add_subplot(gs[0, 0])
    axW2 = fig.add_subplot(gs[1, 0])
    axH  = fig.add_subplot(gs[0, 1])
    axO  = fig.add_subplot(gs[1, 1])
    axS  = fig.add_subplot(gs[:, 2])

    # --- static: encoded weight tiles ---
    wlim1 = np.abs(W1).max()
    axW1.imshow(W1, cmap="PuOr", vmin=-wlim1, vmax=wlim1, aspect="auto")
    axW1.set_title("Tile 1 — input→hidden\n(weights stored as conductance)", fontsize=9)
    axW1.set_xlabel("hidden unit (bitline)", fontsize=8)
    axW1.set_ylabel("input\n(wordline)", fontsize=8)
    axW1.set_yticks([0, 1, 2]); axW1.set_yticklabels(["x₁", "x₂", "bias"], fontsize=8)

    wlim2 = np.abs(W2).max()
    axW2.imshow(W2.T, cmap="PuOr", vmin=-wlim2, vmax=wlim2, aspect="auto")
    axW2.set_title("Tile 2 — hidden→class", fontsize=9)
    axW2.set_xlabel("hidden unit (wordline)", fontsize=8)
    axW2.set_yticks([0, 1]); axW2.set_yticklabels(["A", "B"], fontsize=8)
    axW2.set_ylabel("class\n(bitline)", fontsize=8)

    # --- dynamic: hidden activations (8x8) ---
    him = axH.imshow(np.zeros((8, 8)), cmap="RdBu_r", vmin=-1, vmax=1)
    axH.set_title("Hidden neurons firing (this input)", fontsize=9)
    axH.set_xticks([]); axH.set_yticks([])

    # --- dynamic: output class currents ---
    obars = axO.bar([0, 1], [0, 0], color=[OR, BL])
    axO.set_ylim(0, 1); axO.set_xticks([0, 1]); axO.set_xticklabels(["class A", "class B"], fontsize=9)
    axO.set_ylabel("output (softmax)", fontsize=8)
    axO.set_title("Class currents → decision", fontsize=9)
    otxt = axO.text(0.5, 0.9, "", transform=axO.transAxes, ha="center", fontsize=11, weight="bold")

    # --- static background + dynamic points on the spiral plane ---
    axS.contourf(GX, GY, PB, levels=24, cmap="RdBu", alpha=.45)
    axS.scatter(*X[y == 0].T, s=4, c=OR, alpha=.12)
    axS.scatter(*X[y == 1].T, s=4, c=BL, alpha=.12)
    axS.set_title("Segmenting the spiral — points classified by the array", fontsize=10)
    axS.set_xlabel("x₁"); axS.set_ylabel("x₂"); axS.set_aspect("equal")
    axS.set_xlim(gx.min(), gx.max()); axS.set_ylim(gy.min(), gy.max())
    acc_txt = axS.text(0.02, 0.97, "", transform=axS.transAxes, va="top", fontsize=10,
                       bbox=dict(boxstyle="round", fc="white", alpha=.8))
    star = axS.plot([], [], marker="*", ms=22, mfc="yellow", mec="k", mew=1.2, zorder=5)[0]

    n_correct = [0]

    def update(i):
        xa = Xta[i]
        _, a1, a1a, p = forward(xa[None], W1, W2)
        a1, p = a1[0], p[0]
        pred = int(p.argmax())
        true = int(yt[i])
        ok = pred == true
        n_correct[0] += ok

        him.set_data(a1.reshape(8, 8))
        for k, bar in enumerate(obars):
            bar.set_height(p[k])
        otxt.set_text(f"→ class {'A' if pred == 0 else 'B'}  ({p[pred]*100:.0f}%)")
        otxt.set_color(OR if pred == 0 else BL)

        col = OR if pred == 0 else BL
        if ok:
            axS.scatter([xa[0]], [xa[1]], s=42, c=col, edgecolors="k", linewidths=.4, zorder=4)
        else:
            axS.scatter([xa[0]], [xa[1]], s=70, c="k", marker="x", linewidths=1.8, zorder=4)
        star.set_data([xa[0]], [xa[1]])
        acc_txt.set_text(f"inferred {i+1}/{len(Xta)}\nlive accuracy {100*n_correct[0]/(i+1):.0f}%")
        return him, star

    print("  rendering animation ...")
    anim = FuncAnimation(fig, update, frames=len(Xta), interval=90, blit=False)
    anim.save("crossbar_inference.gif", writer=PillowWriter(fps=11), dpi=90)
    star.set_data([], [])
    fig.savefig("crossbar_inference_final.png", dpi=110)
    plt.close(fig)
    print("  wrote crossbar_inference.gif and crossbar_inference_final.png")


if __name__ == "__main__":
    print("Crossbar test-time inference visualisation")
    print("-" * 44)
    main()
    print("-" * 44)
    print("done.")
