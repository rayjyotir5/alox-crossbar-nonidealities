"""
Multi-frame (filmstrip) versions of the time-varying visualizations, for the
paper (clearer than a single still). Produces:
  - insitu_inference_strip.png : test-time inference segmenting the spiral, in 4 stages
  - retention_boundary_strip.png : decision boundary dissolving as conductance drifts
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crossbar_learn import make_spirals, train_crossbar, forward, accuracy
from crossbar_advanced import drift_state

OR, BL = "#d9821a", "#2a7fb8"


def train_net():
    rng = np.random.default_rng(0)
    X, y = make_spirals(n_per=500, turns=1.5, noise=0.08, rng=rng)
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    Y = np.eye(2)[y]
    _, (t1, t2) = train_crossbar(Xa, Y, y, 64, 10.0, 350, np.random.default_rng(10),
                                 False, 1.5e-3, 3e-3, 32, lr_decay=0.992)
    return X, y, t1, t2


def grid_of(X):
    gx = np.linspace(X[:, 0].min() - .25, X[:, 0].max() + .25, 200)
    gy = np.linspace(X[:, 1].min() - .25, X[:, 1].max() + .25, 200)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.stack([GX.ravel(), GY.ravel(), np.ones(GX.size)], 1)
    return gx, gy, GX, GY, grid


def inference_strip(X, y, t1, t2):
    W1, W2 = t1.weight(), t2.weight()
    gx, gy, GX, GY, grid = grid_of(X)
    PB = forward(grid, W1, W2)[3][:, 1].reshape(GX.shape)
    # fresh test spiral, revealed centre-outward
    Xt, yt = make_spirals(n_per=80, turns=1.5, noise=0.06, rng=np.random.default_rng(99))
    order = np.argsort(np.hypot(Xt[:, 0], Xt[:, 1]))
    Xt, yt = Xt[order], yt[order]
    Xta = np.concatenate([Xt, np.ones((len(Xt), 1))], 1)
    pred = forward(Xta, W1, W2)[3].argmax(1)

    cuts = [0.25, 0.5, 0.75, 1.0]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for k, c in enumerate(cuts):
        n = int(c * len(Xta))
        a = ax[k]
        a.contourf(GX, GY, PB, levels=20, cmap="RdBu", alpha=.45)
        a.scatter(*X[y == 0].T, s=3, c=OR, alpha=.10)
        a.scatter(*X[y == 1].T, s=3, c=BL, alpha=.10)
        ok = pred[:n] == yt[:n]
        pc = pred[:n]
        for cls, col in [(0, OR), (1, BL)]:
            m = ok & (pc == cls)
            a.scatter(Xt[:n][m, 0], Xt[:n][m, 1], s=22, c=col, edgecolors="k", linewidths=.3, zorder=4)
        bad = ~ok
        a.scatter(Xt[:n][bad, 0], Xt[:n][bad, 1], s=40, c="k", marker="x", linewidths=1.4, zorder=5)
        acc = 100 * np.mean(pred[:n] == yt[:n])
        a.set_title(f"{n}/{len(Xta)} points  ({acc:.0f}%)", fontsize=11)
        a.set_xlim(gx.min(), gx.max()); a.set_ylim(gy.min(), gy.max())
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Test-time inference on the crossbar: the spiral is segmented point by point (left to right)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("insitu_inference_strip.png", dpi=120); plt.close(fig)
    print("wrote insitu_inference_strip.png")


def retention_strip(X, y, t1, t2):
    gx, gy, GX, GY, grid = grid_of(X)
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    w0 = [(t1.Gp.w.copy(), t1.Gm.w.copy()), (t2.Gp.w.copy(), t2.Gm.w.copy())]
    T = 85 + 273.15
    snaps = [(0, "t = 0"), (86400, "1 day"), (7 * 86400, "1 week"), (30 * 86400, "1 month")]
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for k, (t, lab) in enumerate(snaps):
        t1.Gp.w = drift_state(w0[0][0], t, T); t1.Gm.w = drift_state(w0[0][1], t, T)
        t2.Gp.w = drift_state(w0[1][0], t, T); t2.Gm.w = drift_state(w0[1][1], t, T)
        W1, W2 = t1.weight(), t2.weight()
        PB = forward(grid, W1, W2)[3][:, 1].reshape(GX.shape)
        acc = accuracy(Xa, W1, W2, y) * 100
        a = ax[k]
        a.contourf(GX, GY, PB, levels=20, cmap="RdBu", alpha=.55, vmin=0, vmax=1)
        a.scatter(*X[y == 0].T, s=4, c=OR, alpha=.30)
        a.scatter(*X[y == 1].T, s=4, c=BL, alpha=.30)
        a.set_title(f"{lab}  ({acc:.0f}%)", fontsize=11)
        a.set_xlim(gx.min(), gx.max()); a.set_ylim(gy.min(), gy.max())
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    for tl, (gp, gm) in zip((t1, t2), w0):
        tl.Gp.w, tl.Gm.w = gp, gm
    fig.suptitle("Retention drift at 85 C: the trained decision boundary dissolves to chance over a month", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig("retention_boundary_strip.png", dpi=120); plt.close(fig)
    print("wrote retention_boundary_strip.png")


if __name__ == "__main__":
    X, y, t1, t2 = train_net()
    inference_strip(X, y, t1, t2)
    retention_strip(X, y, t1, t2)
