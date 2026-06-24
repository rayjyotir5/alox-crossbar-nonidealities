"""
Why the crossbar is fast: N×M sequential MACs  vs  ONE parallel read
====================================================================

Same matrix-vector product  y = Wx,  computed two ways, side by side:

  LEFT  — digital / von Neumann: one multiply-accumulate per clock cycle. The
          processor fetches W_ij and x_i from memory, multiplies, adds to an
          accumulator, and repeats — N×M times. Data shuttles across the
          memory bus every step (the von Neumann bottleneck). A cycle counter
          climbs.

  RIGHT — analog crossbar / in-memory: all inputs are applied as wordline
          voltages at once; EVERY device multiplies simultaneously (Ohm's law,
          I=V·G) and EVERY column sums its currents simultaneously (Kirchhoff).
          The whole product is read out in ONE step — the counter says 1.

Watch the digital counter race to N×M while the crossbar already holds the
answer. That gap is the whole efficiency argument.

Run:  python3 crossbar_parallel_viz.py   ->   crossbar_parallel.gif (+ PNG)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter

RED, GRN, GLD = "#c0392b", "#1e8449", "#e8a13a"
N, M = 8, 6                                     # 8 inputs (wordlines), 6 outputs (bitlines)


def main():
    rng = np.random.default_rng(7)
    W = np.round(rng.uniform(-1, 1, (N, M)), 2)
    x = np.round(rng.uniform(0.1, 1.0, N), 2)
    contrib = x[:, None] * W                     # per-device product x_i*W_ij
    y = contrib.sum(0)                           # the answer

    # digital: cumulative column sums in raster order (one cell per cycle)
    order = [(s // M, s % M) for s in range(N * M)]
    partial = np.zeros((N * M, M))
    run = np.zeros(M)
    for s, (i, j) in enumerate(order):
        run[j] += contrib[i, j]
        partial[s] = run
    n_cells = N * M

    frames = n_cells + 18                        # + hold at the end
    wlim = np.abs(W).max()
    ylim = (min(y.min(), 0) * 1.25 - .1, max(y.max(), 0) * 1.25 + .1)

    fig = plt.figure(figsize=(14, 7.4))
    fig.suptitle("Same computation  y = W·x  —  digital does N×M steps,  "
                 "the crossbar does it in ONE read", fontsize=13, y=0.97)
    gs = GridSpec(2, 2, height_ratios=[2.4, 1], hspace=.33, wspace=.18,
                  left=.06, right=.97, top=.86, bottom=.07)
    axDg = fig.add_subplot(gs[0, 0]); axCg = fig.add_subplot(gs[0, 1])
    axDb = fig.add_subplot(gs[1, 0]); axCb = fig.add_subplot(gs[1, 1])

    # ---------- grids (identical weight matrix on both sides) ----------
    for ax, title in [(axDg, "DIGITAL  (von Neumann)"), (axCg, "CROSSBAR  (in-memory)")]:
        ax.imshow(W, cmap="PuOr", vmin=-wlim, vmax=wlim, aspect="equal")
        ax.set_xticks(range(M)); ax.set_yticks(range(N))
        ax.set_xticklabels([f"y{j}" for j in range(M)], fontsize=8)
        ax.set_yticklabels([f"x{i}={x[i]:.2f}" for i in range(N)], fontsize=8)
        ax.set_title(title, fontsize=11, pad=8)
        for i in range(N):
            for j in range(M):
                ax.text(j, i, f"{W[i, j]:+.1f}", ha="center", va="center",
                        fontsize=7, color="#333")

    # digital: a single moving "ALU" highlight
    hl = Rectangle((-.5, -.5), 1, 1, fill=False, ec=RED, lw=3, zorder=5)
    axDg.add_patch(hl)
    dg_counter = axDg.text(0.5, -0.13, "", transform=axDg.transAxes, ha="center",
                           fontsize=13, weight="bold", color=RED)
    bus = axDg.text(0.5, 1.07, "", transform=axDg.transAxes, ha="center",
                    fontsize=9, color="#666")

    # crossbar: input voltage strip + a glow overlay (all cells active at once)
    glow = axCg.imshow(np.abs(contrib), cmap="hot", alpha=0.0, aspect="equal",
                       vmin=0, vmax=np.abs(contrib).max())
    cg_counter = axCg.text(0.5, -0.13, "", transform=axCg.transAxes, ha="center",
                           fontsize=13, weight="bold", color=GRN)
    axCg.text(-0.06, 0.5, "all wordlines\ndriven at once →", transform=axCg.transAxes,
              ha="right", va="center", fontsize=8, color=GRN, rotation=0)

    # ---------- output bars ----------
    dbars = axDb.bar(range(M), np.zeros(M), color="#bbb", ec="#777")
    axDb.set_ylim(*ylim); axDb.set_xticks(range(M))
    axDb.set_xticklabels([f"y{j}" for j in range(M)], fontsize=8)
    axDb.set_title("output accumulators — filling one cell at a time", fontsize=9)
    axDb.axhline(0, color="k", lw=.5); axDb.grid(alpha=.25, axis="y")

    cbars = axCb.bar(range(M), y, color=GRN, ec="#145a32")
    axCb.set_ylim(*ylim); axCb.set_xticks(range(M))
    axCb.set_xticklabels([f"y{j}" for j in range(M)], fontsize=8)
    axCb.set_title("bitline currents = full output, ready after one read", fontsize=9)
    axCb.axhline(0, color="k", lw=.5); axCb.grid(alpha=.25, axis="y")

    banner = fig.text(0.5, 0.005, "", ha="center", fontsize=10.5, color="#222")

    def update(k):
        s = min(k, n_cells - 1)
        i, j = order[s]
        done_digital = k >= n_cells
        # digital highlight + counter + bus blink
        hl.set_xy((j - .5, i - .5))
        hl.set_visible(not done_digital)
        cyc = n_cells if done_digital else k + 1
        dg_counter.set_text(f"cycle {cyc} / {n_cells}  ({100*cyc//n_cells}% done)")
        bus.set_text("memory → ALU  (fetch xᵢ, Wᵢⱼ)" if (k % 2 == 0 and not done_digital)
                     else " ")
        for jj, b in enumerate(dbars):
            b.set_height(partial[s, jj])
            b.set_color(GRN if done_digital else "#bbb")

        # crossbar: lights up at step 1 and stays done
        on = min(1.0, k / 2.0)
        glow.set_alpha(0.55 * on)
        cg_counter.set_text("1 read window  ·  ✓ DONE" if k >= 1 else "applying voltages…")

        if done_digital:
            banner.set_text("Crossbar finished at step 1.  Digital needed "
                            f"{n_cells} sequential MACs + a memory fetch each step — "
                            "the crossbar did all 48 multiplies and 6 column-sums in one parallel physical step.")
        else:
            banner.set_text(f"Digital: multiply x{i}·W[{i},{j}], add to y{j}, fetch next …      "
                            "Crossbar: every device already multiplied; every column already summed.")
        return [hl, glow]

    print("  rendering ...")
    anim = FuncAnimation(fig, update, frames=frames, interval=110, blit=False)
    anim.save("crossbar_parallel.gif", writer=PillowWriter(fps=9), dpi=92)
    update(frames - 1)
    fig.savefig("crossbar_parallel_final.png", dpi=115)
    plt.close(fig)
    print("  wrote crossbar_parallel.gif and crossbar_parallel_final.png")


if __name__ == "__main__":
    print("Crossbar parallelism visualisation: N×M steps vs one read")
    print("-" * 58)
    main()
    print("-" * 58)
    print("done.")
