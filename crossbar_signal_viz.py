"""
Analog signal flow through the crossbar — inputs encoded as waveforms
=====================================================================

A single inference shown in the TIME DOMAIN, so you can see how a numeric input
becomes analog voltage signals and how the output is physically generated.

Encoding: pulse-width modulation (PWM). Each value drives its wordline with a
fixed-amplitude voltage pulse whose WIDTH ∝ |value| and POLARITY = sign(value).
While a pulse is high, that device sources I = V·G onto its bitline, so the
bitline current is a staircase (steps off as pulses end) and the integrated
charge equals the multiply-accumulate  Σ x_i · W_ij  — the VMM result.

The inference runs as two read phases on one timeline:
  phase 1  (t: 0→1)  Tile 1 reads the 2 inputs → 64 hidden pre-activations
  phase 2  (t: 1→2)  Tile 2 reads the 64 hidden values → 2 class outputs

Panels (right, shared time axis): wordline voltages, bitline currents, and the
accumulated charge converging to the MAC result. Left: which tile is active.

Run:  python3 crossbar_signal_viz.py   ->   crossbar_signals.gif (+ final PNG)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.animation import FuncAnimation, PillowWriter

from crossbar_learn import make_spirals, train_crossbar, softmax

OR, BL = "#d9821a", "#2a7fb8"
POS, NEG = "#c0392b", "#2471a3"        # positive / negative pulse polarity
VREAD, VMAX = 0.2, 1.35                 # read amplitude; input full-scale


def pwm(values, vfull):
    """width ∝ |value|/vfull (clamped to [0,1]); height = sign(value)."""
    w = np.clip(np.abs(values) / vfull, 0.0, 1.0)
    return w, np.sign(values)


def main():
    rng = np.random.default_rng(0)
    X, y = make_spirals(n_per=500, turns=1.5, noise=0.08, rng=rng)
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    Y = np.eye(2)[y]
    print("  training crossbar ...")
    acc, (t1, t2) = train_crossbar(Xa, Y, y, 64, 10.0, 400,
                                   np.random.default_rng(5), False, 1.5e-3, 3e-3, 32)
    W1, W2 = t1.weight(), t2.weight()             # (3,64) and (65,2)
    print(f"  trained accuracy {acc[-1]*100:.1f}%")

    # pick a confidently, correctly classified outer-arm test point
    Xt, yt = make_spirals(n_per=80, turns=1.5, noise=0.05, rng=np.random.default_rng(99))
    Xta = np.concatenate([Xt, np.ones((len(Xt), 1))], 1)
    a1_all = np.tanh(Xta @ W1)
    p_all = softmax(np.concatenate([a1_all, np.ones((len(a1_all), 1))], 1) @ W2)
    correct = p_all.argmax(1) == yt
    pick = np.where(correct)[0][np.argmax(p_all[correct].max(1) *
                                          np.hypot(*Xt[correct].T))]
    x = Xta[pick]                                  # [x1, x2, 1]
    true = int(yt[pick])

    # ---- forward, exact ----
    z1 = x @ W1                                    # (64,)
    a1 = np.tanh(z1)
    a1a = np.concatenate([a1, [1.0]])              # (65,)
    z2 = a1a @ W2                                   # (2,)
    p = softmax(z2[None])[0]
    pred = int(p.argmax())

    # ---- PWM encodings ----
    w1, s1 = pwm(x, VMAX)                           # 3 inputs
    w2, s2 = pwm(a1a, 1.0)                           # 65 hidden+bias inputs
    sel = np.argsort(-np.abs(z1))[:3]               # 3 most-active hidden bitlines

    # ---- precompute waveforms over the 2-phase timeline ----
    nF = 240
    ts = np.linspace(0, 2, nF)
    Vwl = np.full((nF, 3), np.nan)                  # phase-1 input voltages
    Icur = np.full((nF, 3), np.nan)                 # phase-1 selected hidden currents
    Qch = np.full((nF, 3), np.nan)                  # phase-1 selected hidden charge
    I2 = np.full((nF, 2), np.nan)                   # phase-2 class currents
    Q2 = np.full((nF, 2), np.nan)                   # phase-2 class charge
    h1 = s1[:, None] * W1                            # (3,64) step heights
    h2 = s2[:, None] * W2                            # (65,2) step heights
    for k, t in enumerate(ts):
        if t <= 1.0:                                 # phase 1
            lo = t
            on = lo < w1
            Vwl[k] = np.where(on, s1 * VREAD, 0.0)
            Icur[k] = (np.where(lo < w1, 1.0, 0.0)[:, None] * h1[:, sel]).sum(0)
            Qch[k] = (np.minimum(lo, w1)[:, None] * h1[:, sel]).sum(0) * VMAX
        else:                                        # phase 2
            lo = t - 1.0
            I2[k] = (np.where(lo < w2, 1.0, 0.0)[:, None] * h2).sum(0)
            Q2[k] = (np.minimum(lo, w2)[:, None] * h2).sum(0)

    # ============================ figure ============================
    fig = plt.figure(figsize=(14, 7.6))
    fig.suptitle("Analog inference in the time domain — inputs encoded as "
                 "pulse-width signals, output integrated on the bitlines", fontsize=12.5)
    gs = GridSpec(3, 2, width_ratios=[1, 2.6], hspace=.35, wspace=.16,
                  left=.04, right=.985, top=.91, bottom=.08)
    axSch = fig.add_subplot(gs[:, 0]); axSch.axis("off")
    axV = fig.add_subplot(gs[0, 1])
    axI = fig.add_subplot(gs[1, 1], sharex=axV)
    axQ = fig.add_subplot(gs[2, 1], sharex=axV)

    # ---- left: schematic of the two tiles ----
    def tile(xc, yc, label, w=0.34, h=0.2):
        box = FancyBboxPatch((xc - w / 2, yc - h / 2), w, h,
                             boxstyle="round,pad=0.02", mutation_aspect=1,
                             ec="#333", fc="#eef2f6", lw=1.4)
        axSch.add_patch(box)
        axSch.text(xc, yc, label, ha="center", va="center", fontsize=9)
        return box
    box1 = tile(0.5, 0.72, "Tile 1\n3 × 64\n(input→hidden)")
    box2 = tile(0.5, 0.34, "Tile 2\n65 × 2\n(hidden→class)")
    axSch.annotate("", (0.5, 0.62), (0.5, 0.44),
                   arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#666"))
    axSch.text(0.5, 0.53, "tanh", ha="center", fontsize=8, color="#666",
               bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    axSch.annotate("inputs", (0.5, 0.86), (0.5, 0.95), ha="center", fontsize=8,
                   arrowprops=dict(arrowstyle="-|>", lw=1.4, color=OR))
    axSch.annotate("", (0.5, 0.14), (0.5, 0.24),
                   arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#666"))
    ptxt = axSch.text(0.5, 0.06, "", ha="center", fontsize=10, weight="bold")
    axSch.text(0.5, 0.99, f"input  (x₁,x₂) = ({x[0]:+.2f}, {x[1]:+.2f})",
               ha="center", fontsize=9)
    axSch.set_xlim(0, 1); axSch.set_ylim(0, 1)

    # ---- right: waveform panels ----
    lV = [axV.plot([], [], lw=1.6, color=c, label=lab)[0]
          for c, lab in [(OR, "x₁"), (BL, "x₂"), ("#888", "bias")]]
    axV.set_ylabel("wordline V"); axV.set_ylim(-VREAD * 1.3, VREAD * 1.3)
    axV.axhline(0, color="k", lw=.4); axV.legend(ncol=3, fontsize=8, loc="upper right")
    axV.set_title("phase 1: input voltages (PWM)        phase 2: hidden-vector pulses drive Tile 2",
                  fontsize=9)

    # phase-2 hidden raster drawn in axV's upper region (compact)
    for j in range(64):
        axV.add_patch(plt.Rectangle((1.0, VREAD * (0.15 + 0.012 * j) - VREAD * 0.0),
                                    w2[j], VREAD * 0.011,
                                    color=(POS if a1[j] >= 0 else NEG),
                                    alpha=0.25 + 0.6 * min(abs(a1[j]), 1)))
    lI = [axI.plot([], [], lw=1.4, color=c)[0] for c in ["#7a5195", "#ef5675", "#ffa600"]]
    lI2 = [axI.plot([], [], lw=2.0, color=c)[0] for c in [OR, BL]]
    axI.set_ylabel("bitline I (a.u.)"); axI.axhline(0, color="k", lw=.4)
    axI.set_title("bitline currents — sum of active device currents (a staircase)", fontsize=9)

    lQ = [axQ.plot([], [], lw=1.4, color=c)[0] for c in ["#7a5195", "#ef5675", "#ffa600"]]
    lQ2 = [axQ.plot([], [], lw=2.2, color=c)[0] for c in [OR, BL]]
    for jj, j in enumerate(sel):
        axQ.axhline(z1[j], ls=":", lw=.7, color=["#7a5195", "#ef5675", "#ffa600"][jj])
    axQ.axhline(z2[0], ls=":", lw=.7, color=OR); axQ.axhline(z2[1], ls=":", lw=.7, color=BL)
    axQ.set_ylabel("accumulated\ncharge = MAC"); axQ.set_xlabel("time (two read phases)")
    axQ.axhline(0, color="k", lw=.4)
    axQ.set_title("accumulated charge converges to the dot product  Σ xᵢ·Wᵢⱼ", fontsize=9)

    cursors = [ax.axvline(0, color="k", lw=1.1, alpha=.6) for ax in (axV, axI, axQ)]
    for ax in (axV, axI, axQ):
        ax.axvline(1.0, color="#999", ls="--", lw=.8)
        ax.set_xlim(0, 2); ax.grid(alpha=.25)

    def update(k):
        for d in range(3):
            lV[d].set_data(ts[:k + 1], Vwl[:k + 1, d])
            lI[d].set_data(ts[:k + 1], Icur[:k + 1, d])
            lQ[d].set_data(ts[:k + 1], Qch[:k + 1, d])
        for c in range(2):
            lI2[c].set_data(ts[:k + 1], I2[:k + 1, c])
            lQ2[c].set_data(ts[:k + 1], Q2[:k + 1, c])
        for cur in cursors:
            cur.set_xdata([ts[k], ts[k]])
        active = ts[k] <= 1.0
        box1.set_facecolor("#ffe9c9" if active else "#eef2f6")
        box2.set_facecolor("#eef2f6" if active else "#d9ecff")
        if ts[k] > 1.5:
            ptxt.set_text(f"→ class {'A' if pred == 0 else 'B'}  ({p[pred]*100:.0f}%)")
            ptxt.set_color(OR if pred == 0 else BL)
        else:
            ptxt.set_text("")
        return lV + lI + lQ + lI2 + lQ2

    # auto-scale current/charge panels to the data
    allI = np.concatenate([Icur, I2], axis=1)
    axI.set_ylim(np.nanmin(allI) * 1.15, np.nanmax(allI) * 1.15)
    qlo = min(z1[sel].min(), z2.min()); qhi = max(z1[sel].max(), z2.max())
    axQ.set_ylim(qlo - 0.2 * abs(qlo) - 0.1, qhi + 0.25 * abs(qhi) + 0.1)

    print("  rendering ...")
    anim = FuncAnimation(fig, update, frames=nF, interval=60, blit=False)
    anim.save("crossbar_signals.gif", writer=PillowWriter(fps=22), dpi=88)
    update(nF - 1)
    fig.savefig("crossbar_signals_final.png", dpi=115)
    plt.close(fig)
    print(f"  decision: class {'A' if pred==0 else 'B'} (true {'A' if true==0 else 'B'})")
    print("  wrote crossbar_signals.gif and crossbar_signals_final.png")


if __name__ == "__main__":
    print("Crossbar analog-signal (waveform) visualisation")
    print("-" * 48)
    main()
    print("-" * 48)
    print("done.")
