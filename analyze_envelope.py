"""
Analyze the robotics inference-mapping sweep: real-accuracy operating envelope
(1R vs 1T1R), out-of-sample predictive validation (cheap MVM-error feature
predicts real task accuracy on held-out configs), and the ADC/DAC ablation.
Reads results_*.jsonl, writes figures to ../figures/, prints key numbers.
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = []
for f in glob.glob("results_*.jsonl"):
    for ln in open(f):
        ln = ln.strip()
        if ln:
            rows.append(json.loads(ln))
print(f"loaded {len(rows)} configs")
FLOAT = rows[0]["acc_sw"]
grid = [r for r in rows if r["kind"] == "grid"]
Ns = sorted({r["N"] for r in grid})
rws = sorted({r["r_w"] for r in grid})


def acc_at(cell, N, rw):
    for r in grid:
        if r["cell"] == cell and r["N"] == N and r["r_w"] == rw:
            return r["acc"]
    return np.nan


def usable_N(cell, rw=2.0, margin=2.0):
    ok = [N for N in Ns if acc_at(cell, N, rw) >= FLOAT - margin]
    return max(ok) if ok else 0


def oos(rs, seed=0):
    x = np.log10(np.array([r["mvm_err"] for r in rs]) + 0.1)
    y = np.array([r["acc"] for r in rs])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(rs)); cut = int(0.6 * len(rs))
    tr, te = idx[:cut], idx[cut:]
    c = np.polyfit(x[tr], y[tr], 2)
    pred = np.polyval(c, x[te])
    ss = np.sum((y[te] - pred) ** 2); st = np.sum((y[te] - y[te].mean()) ** 2)
    return 1 - ss / (st + 1e-12), np.mean(np.abs(y[te] - pred)), np.corrcoef(x, y)[0, 1], \
        (x, y, te, pred, c)


g1R = [r for r in grid if r["cell"] == "1R"]
g1T = [r for r in grid if r["cell"] == "1T1R"]
r2_1R, mae_1R, pear_1R, V = oos(g1R)
r2_pool, mae_pool, pear_pool, _ = oos(grid)
print(f"[OOS within 1R]  Pearson={pear_1R:.2f}  held-out R2={r2_1R:.2f}  MAE={mae_1R:.1f} pts")
print(f"[OOS pooled   ]  Pearson={pear_pool:.2f}  held-out R2={r2_pool:.2f}  MAE={mae_pool:.1f} pts")

# (a) envelope heatmaps
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for a, cell in zip(ax, ("1R", "1T1R")):
    Z = np.array([[acc_at(cell, N, rw) for N in Ns] for rw in rws])
    cf = a.contourf(Ns, rws, Z, levels=np.linspace(0, 100, 21), cmap="RdYlGn")
    cs = a.contour(Ns, rws, Z, levels=[FLOAT - 2], colors="k", linewidths=1.3)
    a.clabel(cs, fmt="%.0f%%", fontsize=8)
    a.set_yscale("log"); a.set_yticks(rws); a.set_yticklabels([f"{r:.1f}" for r in rws])
    a.set_xlabel("array size N (N x N)"); a.set_title(f"({'a' if cell=='1R' else 'b'}) {cell} cell")
ax[0].set_ylabel(r"wire resistance $r_w$ ($\Omega$/cell)")
fig.colorbar(cf, ax=ax, fraction=0.04, label="within-tol steering accuracy (%)")
fig.suptitle("Robotics steering network: real-accuracy operating envelope "
             f"(float ceiling {FLOAT:.1f}%; black contour = float - 2 pts)", fontsize=11)
fig.savefig("../figures/robotics_envelope.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# (b) out-of-sample validation
x, y, te, pred, c = V
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].scatter([r["mvm_err"] for r in g1R], [r["acc"] for r in g1R], s=28, c="#2a7fb8", label="1R")
ax[0].scatter([r["mvm_err"] for r in g1T], [r["acc"] for r in g1T], s=28, c="#d9821a", label="1T1R")
xs = np.linspace(min(x), max(x), 100)
ax[0].plot(10 ** xs - 0.1, np.polyval(c, xs), "k--", lw=1, label="fit (1R)")
ax[0].set(xlabel="MVM-error feature (%)", ylabel="task accuracy (%)",
          title="(a) Cheap feature vs real accuracy")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[1].scatter(y[te], pred, s=36, c="#2a9d8f")
lim = [min(y[te].min(), pred.min()) - 3, max(y[te].max(), pred.max()) + 3]
ax[1].plot(lim, lim, "k--", lw=.8)
ax[1].set(xlabel="actual held-out accuracy (%)", ylabel="predicted (%)", xlim=lim, ylim=lim,
          title=f"(b) Out-of-sample (1R): R$^2$={r2_1R:.2f}, MAE={mae_1R:.1f} pts")
ax[1].grid(alpha=.3)
fig.suptitle("Out-of-sample validation: cheap MVM-error feature predicts real task accuracy", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig("../figures/robotics_validation.png", dpi=130); plt.close(fig)

# (c) ADC/DAC ablation
adc = sorted([r for r in rows if r["kind"] == "adcdac"], key=lambda r: (r["adc_bits"] or 99))
labels = [("full" if r["adc_bits"] is None else f"{r['adc_bits']}b") for r in adc]
fig, a = plt.subplots(figsize=(6, 4))
a.bar(range(len(adc)), [r["acc"] for r in adc], color="#7a5195")
a.set_xticks(range(len(adc))); a.set_xticklabels(labels)
a.axhline(FLOAT, ls="--", c="grey", label=f"float {FLOAT:.0f}%")
a.set(xlabel="ADC/DAC precision", ylabel="task accuracy (%)",
      title="ADC/DAC ablation (N=96, $r_w$=2, 1R)", ylim=(0, 100))
a.legend(fontsize=8); a.grid(alpha=.3, axis="y")
fig.tight_layout(); fig.savefig("../figures/robotics_adcdac.png", dpi=130); plt.close(fig)

print(f"float accuracy: {FLOAT:.1f}%")
print("1R envelope at r_w=2:", {N: round(acc_at("1R", N, 2.0), 1) for N in Ns})
print("1T1R envelope at r_w=2:", {N: round(acc_at("1T1R", N, 2.0), 1) for N in Ns})
print(f"usable N (within 2 pts of float, r_w=2): 1R={usable_N('1R')}  1T1R={usable_N('1T1R')}")
print("1T1R vs 1R at (N=96,rw=2):", round(acc_at("1T1R", 96, 2.0), 1), "vs", round(acc_at("1R", 96, 2.0), 1))
print("ADC/DAC:", {l: round(r["acc"], 1) for l, r in zip(labels, adc)})
