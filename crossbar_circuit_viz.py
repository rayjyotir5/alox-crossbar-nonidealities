"""
Analog inputs entering the crossbar in parallel
================================================

A real crossbar circuit, animated to show the parallelism explicitly:

  * every wordline (row) has its own analog voltage source on the left; they all
    switch ON at the SAME instant and energise all rows together (gold),
  * each crosspoint memristor then passes I = V·G,
  * current flows DOWN every bitline (column) simultaneously (the moving dots),
  * each column's currents sum (Kirchhoff) into its output at the bottom — all
    columns finishing together.

No stepping, no sequencing: N inputs in, M outputs out, in one read.

Run:  python3 crossbar_circuit_viz.py   ->   crossbar_circuit.gif (+ PNG)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.animation import FuncAnimation, PillowWriter

N, M = 6, 5                                  # 6 wordlines (inputs), 5 bitlines (outputs)
SRC_X = -2.4                                 # x of the voltage sources
GOLD, POS, NEG = "#e8a13a", "#c0392b", "#2471a3"


def main():
    rng = np.random.default_rng(3)
    W = np.round(rng.uniform(-1, 1, (N, M)), 2)
    x = np.round(rng.uniform(0.2, 1.0, N), 2)        # analog input values
    y = x @ W                                         # the answer (column sums)
    ynorm = np.abs(y) / np.abs(y).max()

    rowy = lambda i: (N - 1 - i)                      # row i drawn top→down
    top_y, bot_y = N - 0.4, -1.4                      # bitline extent
    span = top_y - bot_y

    fig = plt.figure(figsize=(12.5, 8.2))
    fig.suptitle("Analog inputs enter the crossbar in PARALLEL — "
                 "all wordlines driven at once, all columns sum at once",
                 fontsize=13, y=0.975)
    gs = GridSpec(2, 1, height_ratios=[5, 1.1], hspace=0.12,
                  left=.13, right=.97, top=.92, bottom=.07)
    ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_aspect("equal")
    axo = fig.add_subplot(gs[1])

    ax.set_xlim(SRC_X - 1.0, M - 0.2)
    ax.set_ylim(bot_y - 0.4, top_y + 0.4)

    # ---- static wires ----
    for i in range(N):                                # wordlines
        ax.plot([SRC_X, M - 0.5], [rowy(i), rowy(i)], color="#cfd6dd", lw=2, zorder=1)
    for j in range(M):                                # bitlines
        ax.plot([j, j], [top_y, bot_y], color="#cfd6dd", lw=2, zorder=1)

    # ---- devices ----
    dev_x = np.repeat(np.arange(M), N)
    dev_y = np.array([rowy(i) for i in range(N)] * M)
    devW = W.T.ravel()
    ax.scatter(dev_x, dev_y, s=320, c=devW, cmap="PuOr", vmin=-1, vmax=1,
               edgecolors="#444", linewidths=1.0, zorder=3)
    ax.text(M / 2 - .5, top_y + 0.25, "memristor at every crosspoint  (colour = weight Gᵢⱼ)",
            ha="center", fontsize=8.5, color="#555")

    # ---- input sources (left) ----
    src_boxes, src_pulses = [], []
    for i in range(N):
        b = Rectangle((SRC_X - 0.7, rowy(i) - 0.22), 0.7, 0.44, fc="#eef2f6",
                      ec="#888", lw=1.2, zorder=4)
        ax.add_patch(b); src_boxes.append(b)
        ax.text(SRC_X - 0.35, rowy(i), f"x{i}\n{x[i]:.2f}", ha="center", va="center",
                fontsize=7.5, zorder=5)
        # little pulse glyph just right of the source
        pl, = ax.plot([], [], color=GOLD, lw=2.2, zorder=5)
        src_pulses.append(pl)
    ax.text(SRC_X - 0.35, top_y + 0.1, "voltage\nsources", ha="center", fontsize=8,
            color="#555", weight="bold")

    # ---- dynamic: wordline energise overlays ----
    wl_hot = [ax.plot([], [], color=GOLD, lw=5, alpha=.8, zorder=2,
                      solid_capstyle="round")[0] for _ in range(N)]

    # ---- dynamic: current dots flowing down every bitline together ----
    K = 5
    base = np.linspace(0, 1, K, endpoint=False)
    dot_x = np.repeat(np.arange(M), K).astype(float)
    dots = ax.scatter(dot_x, np.full(M * K, top_y), s=70, zorder=4)

    # ---- output (bottom) ----
    for j in range(M):
        ax.annotate("", (j, bot_y - 0.05), (j, bot_y + 0.35),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="#1e8449"))
    bars = axo.bar(range(M), np.zeros(M), color="#1e8449", ec="#145a32")
    axo.set_xlim(-0.6, M - 0.4); axo.set_ylim(min(y.min(), 0) * 1.2 - .1, max(y.max(), 0) * 1.2 + .1)
    axo.set_xticks(range(M)); axo.set_xticklabels([f"y{j}" for j in range(M)])
    axo.axhline(0, color="k", lw=.5); axo.grid(alpha=.25, axis="y")
    axo.set_title("bitline outputs  yⱼ = Σᵢ xᵢ·Gᵢⱼ   (all appear together)", fontsize=9.5)

    status = fig.text(0.5, 0.012, "", ha="center",
                      fontsize=11.5, weight="bold", color="#b9770e")

    # phases: A apply+energise [0,0.45]; B current flow [0.4,1.4]; C output fill [0.6,1.4]
    nF = 80
    ts = np.linspace(0, 1.7, nF)

    def update(k):
        t = ts[k]
        a = np.clip(t / 0.45, 0, 1)                       # energise progress (all rows)
        front = SRC_X + a * ((M - 0.5) - SRC_X)
        on = t > 0.02
        for i in range(N):
            src_boxes[i].set_facecolor(GOLD if on else "#eef2f6")
            wl_hot[i].set_data([SRC_X, front], [rowy(i), rowy(i)])
            # source pulse glyph (a small step that appears at t=0)
            if on:
                px = np.array([SRC_X - 0.05, SRC_X - 0.05, SRC_X + 0.18, SRC_X + 0.18])
                ph = rowy(i) + np.array([-0.14, 0.14 * np.sign(x[i] - 0) + 0.0, 0.14, -0.14]) * 0 \
                     + np.array([-0.13, 0.13, 0.13, -0.13])
                src_pulses[i].set_data(px, ph)
            else:
                src_pulses[i].set_data([], [])

        # current flowing down all columns in lockstep once rows are energised
        flow = t > 0.42
        if flow:
            prog = (t - 0.42) * 1.1
            ys = top_y - (((base[None, :] + prog) % 1.0) * span).repeat(M, axis=0).ravel()
            offs = np.c_[dot_x, ys]
            rgba = np.zeros((M * K, 4))
            for j in range(M):
                c = np.array(plt.matplotlib.colors.to_rgb(POS if y[j] >= 0 else NEG))
                rgba[j * K:(j + 1) * K, :3] = c
                rgba[j * K:(j + 1) * K, 3] = 0.2 + 0.78 * ynorm[j]
            dots.set_offsets(offs); dots.set_facecolors(rgba)
            dots.set_sizes(40 + 120 * np.repeat(ynorm, K))
        else:
            dots.set_offsets(np.c_[dot_x, np.full(M * K, top_y)])
            dots.set_facecolors(np.zeros((M * K, 4)))

        o = np.clip((t - 0.6) / 0.6, 0, 1)                # outputs fill together
        for j, b in enumerate(bars):
            b.set_height(y[j] * o)

        if t < 0.45:
            status.set_text("① all 6 voltage sources switch ON together → rows energise")
        elif t < 0.9:
            status.set_text("② every device passes I=V·G → current pours down all 5 columns at once")
        else:
            status.set_text("③ each column's currents sum → all 5 outputs ready — ONE read, fully parallel")
        return wl_hot + [dots] + list(bars)

    print("  rendering ...")
    anim = FuncAnimation(fig, update, frames=nF, interval=90, blit=False)
    anim.save("crossbar_circuit.gif", writer=PillowWriter(fps=11), dpi=92)
    update(int(nF * 0.62))                                # mid-flow frame for the still
    fig.savefig("crossbar_circuit_final.png", dpi=115)
    plt.close(fig)
    print("  wrote crossbar_circuit.gif and crossbar_circuit_final.png")


if __name__ == "__main__":
    print("Crossbar parallel-input circuit visualisation")
    print("-" * 46)
    main()
    print("-" * 46)
    print("done.")
