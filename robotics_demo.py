"""
Demonstration figure: the PilotNet steering network running inference ON the
crossbar. Shows (top) sample road frames with the true steering and the
crossbar-predicted steering at a good vs a degraded operating point, and
(bottom) the steering it commands across a drive, tracking the reference inside
the lane-keeping tolerance at a good point and drifting out of it at a bad point.
Writes ../figures/robotics_demo.png .
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from system_model import infer_pilotnet, TOL

d = np.load("robotics_pilotnet.npz")
w = {k: d[k] for k in d.files if k not in ("Xtest", "ytest", "acc_sw")}
X, y = d["Xtest"], d["ytest"]

base = dict(adc_bits=8, dac_bits=6, read_noise=0.02, ideal=False)
flo = infer_pilotnet(w, X, dict(N=32, r_w=0.5, cell="1R", adc_bits=None,
                                dac_bits=None, read_noise=0.0, ideal=True), np.random.default_rng(0))
good = infer_pilotnet(w, X, dict(N=32, r_w=0.5, cell="1R", **base), np.random.default_rng(1))
bad = infer_pilotnet(w, X, dict(N=128, r_w=4.0, cell="1R", **base), np.random.default_rng(2))


def arrow(ax, s, color, lw=2.5):
    """draw a steering arrow from bottom-centre, angled by steering value."""
    ang = np.clip(s, -0.9, 0.9) * (np.pi / 3)        # steering -> +/- 60 deg
    x0, y0 = 15.5, 30
    ax.plot([x0, x0 + 12 * np.sin(ang)], [y0, y0 - 12 * np.cos(ang)],
            color=color, lw=lw, solid_capstyle="round")


fig = plt.figure(figsize=(14, 6.6))
gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1], hspace=0.32, wspace=0.12)

idx = [3, 18, 33, 51, 70]                            # five varied frames
for k, i in enumerate(idx):
    ax = fig.add_subplot(gs[0, k])
    ax.imshow(X[i, 0], cmap="gray", origin="upper")
    arrow(ax, y[i], "white", 3.5)                    # ground truth
    arrow(ax, good[i], "#2a9d8f")                    # crossbar, good point
    arrow(ax, bad[i], "#c1121f")                     # crossbar, degraded point
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"true {y[i]:+.2f}\nN32 {good[i]:+.2f} / N128 {bad[i]:+.2f}", fontsize=8)
fig.text(0.5, 0.965, "Steering inference on the crossbar: road frame -> commanded steering",
         ha="center", fontsize=12)
fig.text(0.5, 0.50, "white = ground truth,  teal = crossbar at a good point (N=32, $r_w$=0.5),  "
         "red = crossbar at a degraded point (N=128, $r_w$=4)", ha="center", fontsize=9, color="#444")

# bottom: steering trace across a drive
axt = fig.add_subplot(gs[1, :])
n = 80
fr = np.arange(n)
axt.fill_between(fr, y[:n] - TOL, y[:n] + TOL, color="#cfe8e3", label=f"lane-keep tolerance $\\pm${TOL}")
axt.plot(fr, y[:n], "k-", lw=1.6, label="ground truth")
axt.plot(fr, good[:n], color="#2a9d8f", lw=1.4, label="crossbar good (N=32, $r_w$=0.5)")
axt.plot(fr, bad[:n], color="#c1121f", lw=1.4, label="crossbar degraded (N=128, $r_w$=4)")
axt.set(xlabel="frame along the drive", ylabel="steering command", xlim=(0, n - 1))
axt.legend(fontsize=8, ncol=2, loc="upper right"); axt.grid(alpha=.3)
gd = 100 * np.mean(np.abs(good[:n] - y[:n]) < TOL); bd = 100 * np.mean(np.abs(bad[:n] - y[:n]) < TOL)
axt.set_title(f"Commanded steering across the drive: good point stays in-lane "
              f"({gd:.0f}% within tolerance), degraded point drifts out ({bd:.0f}%)", fontsize=10)

fig.savefig("../figures/robotics_demo.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"wrote ../figures/robotics_demo.png  (good {gd:.0f}% vs degraded {bd:.0f}% within tol)")
