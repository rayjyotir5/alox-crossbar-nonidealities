"""
An advanced-robotics task network for inference-only crossbar mapping:
a PilotNet-style end-to-end driving CNN (vision-based lane-following / steering),
trained in floating point on a procedurally generated driving dataset. Saves the
trained weights and a test subset so the crossbar system model evaluates the SAME
network. Steering is framed as a 7-bin classification so "task accuracy" is well
defined.

Run:  python3 robotics_net.py   ->  robotics_pilotnet.npz + printed float accuracy
"""
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
IMG = 32                      # 32x32 grayscale road image
TOL = 0.08                    # steering within-tolerance threshold (task metric)


# ----------------------------------------------------------------------
# procedural driving dataset: road centerline with offset + curvature.
# Continuous steering regression; harder perception (dim, thin road + noise)
# so the network has finite margin and the analog non-idealities can bite.
# ----------------------------------------------------------------------
def gen_driving(n, rng):
    X = np.zeros((n, 1, IMG, IMG), np.float32)
    steer = np.zeros(n, np.float32)
    rows = np.arange(IMG)
    look = ((IMG - 1 - rows) / (IMG - 1)) ** 2        # far rows curve more
    for i in range(n):
        o = rng.uniform(-0.8, 0.8)                    # lateral offset
        c = rng.uniform(-0.8, 0.8)                    # road curvature
        center = IMG / 2 + o * 11 + c * look * 11     # per-row centerline col
        for r in range(IMG):
            d = np.abs(np.arange(IMG) - center[r])
            X[i, 0, r] = 0.7 * np.clip(1.0 - d / 1.8, 0, 1)   # dimmer, thinner band
        X[i, 0] += 0.18 * rng.standard_normal((IMG, IMG))     # heavier sensor noise
        steer[i] = 0.7 * o + 0.3 * c                   # lane-following control
    return np.clip(X, 0, 1), steer


class PilotNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 12, 5, 2, 2)           # 32->16
        self.c2 = nn.Conv2d(12, 24, 5, 2, 2)          # 16->8
        self.c3 = nn.Conv2d(24, 36, 3, 2, 1)          # 8->4
        self.f1 = nn.Linear(36 * 4 * 4, 64)
        self.f2 = nn.Linear(64, 1)        # continuous steering output
        self.r = nn.ReLU()

    def forward(self, x):
        x = self.r(self.c1(x)); x = self.r(self.c2(x)); x = self.r(self.c3(x))
        x = x.flatten(1)
        x = self.r(self.f1(x))
        return self.f2(x)


def main():
    rng = np.random.default_rng(0)
    Xtr, ytr = gen_driving(8000, rng)
    Xte, yte = gen_driving(1500, np.random.default_rng(1))
    net = PilotNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.MSELoss()
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr).unsqueeze(1)
    for ep in range(25):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]
            opt.zero_grad(); lossf(net(Xt[b]), yt[b]).backward(); opt.step()
    with torch.no_grad():
        pred = net(torch.tensor(Xte)).squeeze(1).numpy()
    acc = float(np.mean(np.abs(pred - yte) < TOL))       # within-tolerance accuracy
    print(f"PilotNet steering: within-{TOL} accuracy (float) = {acc*100:.2f}%  "
          f"(MAE {np.mean(np.abs(pred-yte)):.3f})")

    sd = net.state_dict()
    np.savez("robotics_pilotnet.npz",
             c1w=sd["c1.weight"].numpy(), c1b=sd["c1.bias"].numpy(),
             c2w=sd["c2.weight"].numpy(), c2b=sd["c2.bias"].numpy(),
             c3w=sd["c3.weight"].numpy(), c3b=sd["c3.bias"].numpy(),
             f1w=sd["f1.weight"].numpy(), f1b=sd["f1.bias"].numpy(),
             f2w=sd["f2.weight"].numpy(), f2b=sd["f2.bias"].numpy(),
             Xtest=Xte[:200], ytest=yte[:200], acc_sw=acc)
    print("wrote robotics_pilotnet.npz (200 test samples saved)")


if __name__ == "__main__":
    main()
