"""Focused hardening: (1) in-situ training WITH the decay schedule (multi-seed),
(2) an output-calibration baseline for the IR-drop recovery (rules out the
"tuning artifact" concern: can a cheap linear correction match in-situ training?).
"""
import numpy as np
from sklearn.datasets import make_circles
from crossbar_learn import (make_spirals, train_crossbar, train_crossbar_ir,
                            forward, softmax, accuracy, accuracy_ir, CrossbarTile,
                            layer_Gmat, vmm_ir)

def ms(a): a = np.array(a, float); return f"{a.mean():.1f} +/- {a.std():.1f}"

def data(kind, rng):
    if kind == "spirals":
        X, y = make_spirals(n_per=500, turns=1.5, noise=0.08, rng=rng)
    else:
        X, y = make_circles(n_samples=1000, noise=0.10, factor=0.45,
                            random_state=int(rng.integers(0, 9999)))
        X = X * 1.2
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    return X, y, Xa, np.eye(2)[y]

# (1) in-situ with LR decay (apples-to-apples with the IR-aware variant)
def insitu_decay(kind, n=5):
    cv = []
    for s in range(n):
        rng = np.random.default_rng(500 + s)
        X, y, Xa, Y = data(kind, rng)
        acc, _ = train_crossbar(Xa, Y, y, 64, 10.0, 350, np.random.default_rng(10 + s),
                                False, 1.5e-3, 3e-3, 32, lr_decay=0.992)
        cv.append(np.mean(acc[-20:]) * 100)
    print(f"  in-situ (decay schedule), {kind}: {ms(cv)}")

# (2) output-calibration baseline vs in-situ IR-aware training
def calib_vs_train(n=3, rw=1.0):
    naive, calib = [], []
    for s in range(n):
        rng = np.random.default_rng(600 + s)
        X, y, Xa, Y = data("spirals", rng)
        # train an ideal-forward network (decay schedule)
        acc, (t1, t2) = train_crossbar(Xa, Y, y, 64, 10.0, 350,
                                       np.random.default_rng(10 + s), False,
                                       1.5e-3, 3e-3, 32, lr_decay=0.992)
        # IR-distorted vs ideal logits
        z_ir = forward_ir_logits(Xa, t1, t2, rw)
        _, _, _, p_id = forward(Xa, t1.weight(), t2.weight())
        z_id = np.log(np.clip(p_id, 1e-9, 1))      # ideal logit proxy
        naive.append(np.mean(z_ir.argmax(1) == y) * 100)
        # fit a 2x2 affine z_ir -> z_id on half the points, evaluate on the other half
        half = len(y) // 2
        A = np.concatenate([z_ir[:half], np.ones((half, 1))], 1)
        coef, *_ = np.linalg.lstsq(A, z_id[:half], rcond=None)
        Ae = np.concatenate([z_ir[half:], np.ones((len(y) - half, 1))], 1)
        z_cal = Ae @ coef
        calib.append(np.mean(z_cal.argmax(1) == y[half:]) * 100)
    print(f"  IR naive deploy            : {ms(naive)}")
    print(f"  + output affine calibration: {ms(calib)}")
    print(f"  (IR-aware in-situ training : 92.4 +/- 0.9, from seeds_experiments)")

def forward_ir_logits(Xa, t1, t2, rw):
    z1 = vmm_ir(layer_Gmat(t1), Xa, rw) * t1.w_scale
    a1 = np.tanh(z1)
    a1a = np.concatenate([a1, np.ones((len(a1), 1))], 1)
    return vmm_ir(layer_Gmat(t2), a1a, rw) * t2.w_scale

if __name__ == "__main__":
    print("=== in-situ training WITH decay schedule (multi-seed) ===")
    insitu_decay("spirals", 5)
    insitu_decay("circles", 5)
    print("=== IR-drop recovery: calibration baseline vs in-situ training ===")
    calib_vs_train(3)
