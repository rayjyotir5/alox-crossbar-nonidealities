"""
Hardening experiments for the paper: multi-seed error bars, mixed-state IR-drop,
Arrhenius E_a sensitivity, and a second nonlinear task (concentric circles).

Run:  python3 seeds_experiments.py
Prints mean +/- std for every headline number.
"""
import numpy as np
from sklearn.datasets import load_digits, make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from crossbar_advanced import (variable_memristor, read_after, deploy_weights,
                               CrossbarIR, VREAD, drift_state, KB_EV)
from crossbar_learn import (make_spirals, train_crossbar, train_crossbar_ir,
                            forward, softmax, accuracy, accuracy_ir)

def ms(a):
    a = np.array(a, float); return f"{a.mean():.1f} +/- {a.std():.1f}"


# ----------------------------------------------------------------------
# R1a: digit classifier accuracy across seeds (variability + program-verify + IR)
# ----------------------------------------------------------------------
def classifier_seeds(n_seed=5):
    digits = load_digits()
    X = digits.images.reshape(len(digits.images), -1) / 16.0
    y = digits.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=0)
    clf = LogisticRegression(max_iter=2000, C=5.0).fit(Xtr, ytr)
    W, b = clf.coef_.T, clf.intercept_
    acc_sw = clf.score(Xte, yte)
    G_t, dscale = deploy_weights(W)
    Vin = Xte * VREAD

    def acc(G, ir=False, rw=2.5):
        I = CrossbarIR(G, rw).vmm(Vin) if ir else Vin @ G
        logits = (I[:, 0::2] - I[:, 1::2]) / (dscale * VREAD) + b
        return np.mean(logits.argmax(1) == yte)

    openl, verif, withir = [], [], []
    for s in range(n_seed):
        Go = read_after("open",   G_t, np.random.default_rng(100 + s))
        Gv = read_after("verify", G_t, np.random.default_rng(200 + s))
        openl.append(acc(Go) * 100)
        verif.append(acc(Gv) * 100)
        withir.append(acc(Gv, ir=True) * 100)
    print(f"[R1a] digit classifier ({n_seed} seeds)")
    print(f"      software (float)         : {acc_sw*100:.1f}")
    print(f"      open-loop write          : {ms(openl)}")
    print(f"      program-verify write     : {ms(verif)}")
    print(f"      + IR-drop (r=2.5 ohm)    : {ms(withir)}")


# ----------------------------------------------------------------------
# R2: IR-drop error, all-LRS worst case vs realistic mixed-state
# ----------------------------------------------------------------------
def irdrop_mixed(sizes=(16, 32, 64, 128), rw=2.5, n_seed=5):
    print(f"[R2] IR-drop MVM error %, r_w={rw} ohm ({n_seed} seeds)")
    for n in sizes:
        lrs, mix = [], []
        for s in range(n_seed):
            rng = np.random.default_rng(300 + s)
            Vin = np.full(n, VREAD)
            # all-LRS worst case
            G1 = np.full((n, n), 1.0 / 2.0e3) * np.exp(0.15 * rng.standard_normal((n, n)))
            id1 = Vin @ G1; I1 = CrossbarIR(G1, rw).vmm(Vin)[0]
            lrs.append(100 * np.mean(np.abs(I1 - id1) / id1))
            # realistic mixed state: conductances uniform in the usable window
            G2 = rng.uniform(20e-6, 300e-6, (n, n))
            id2 = Vin @ G2; I2 = CrossbarIR(G2, rw).vmm(Vin)[0]
            mix.append(100 * np.mean(np.abs(I2 - id2) / id2))
        print(f"      N={n:4d}  all-LRS: {ms(lrs):>14s}   mixed-state: {ms(mix):>14s}")


# ----------------------------------------------------------------------
# R3: retention Arrhenius E_a sensitivity (time constant at 85 C)
# ----------------------------------------------------------------------
def retention_ea():
    print("[R3] retention time-constant tau at 85 C vs activation energy")
    T = 85 + 273.15
    for Ea in (0.8, 1.0, 1.2):
        # tau0 re-fitted so 25 C retention stays ~10 yr for each Ea (fair comparison)
        tau0 = 3.15e8 / np.exp(Ea / (KB_EV * (25 + 273.15)))
        tau = tau0 * np.exp(Ea / (KB_EV * T))
        print(f"      Ea={Ea:.1f} eV : tau(85C) = {tau:.2e} s = {tau/86400:.2f} days")


# ----------------------------------------------------------------------
# R1b + R4: in-situ learning across seeds, on spirals AND circles
# ----------------------------------------------------------------------
def task_data(kind, rng):
    if kind == "spirals":
        X, y = make_spirals(n_per=500, turns=1.5, noise=0.08, rng=rng)
    else:  # concentric circles (second nonlinear task)
        seed = int(rng.integers(0, 10000))
        X, y = make_circles(n_samples=1000, noise=0.10, factor=0.45, random_state=seed)
        X = X * 1.2
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    return X, y, Xa, np.eye(2)[y]


def insitu_seeds(kind, n_seed=4, n_ir=3):
    print(f"[R1b/R4] in-situ learning on {kind} (in-situ {n_seed} seeds, IR {n_ir} seeds)")
    cv, naive, irtr = [], [], []
    for s in range(n_seed):
        rng = np.random.default_rng(400 + s)
        X, y, Xa, Y = task_data(kind, rng)
        acc_cv, (t1, t2) = train_crossbar(Xa, Y, y, 64, 10.0, 400,
                                          np.random.default_rng(10 + s), False, 1.5e-3, 3e-3, 32)
        cv.append(acc_cv[-1] * 100)
        if s < n_ir:                      # IR experiments are the slow ones
            naive.append(accuracy_ir(Xa, t1, t2, 1.0, y) * 100)
            acc_ir, _ = train_crossbar_ir(Xa, Y, y, 64, 10.0, 350,
                                          np.random.default_rng(10 + s), 1.5e-3, 3e-3, 32, 1.0)
            irtr.append(np.mean(acc_ir[-20:]) * 100)
    print(f"      crossbar in-situ (ideal VMM) : {ms(cv)}")
    print(f"      naive deploy under IR-drop   : {ms(naive)}")
    print(f"      trained with IR-drop in loop : {ms(irtr)}")


if __name__ == "__main__":
    print("=" * 60)
    classifier_seeds(5)
    print("-" * 60); irdrop_mixed(n_seed=5)
    print("-" * 60); retention_ea()
    print("-" * 60); insitu_seeds("spirals", n_seed=4, n_ir=3)
    print("-" * 60); insitu_seeds("circles", n_seed=4, n_ir=3)
    print("=" * 60)
