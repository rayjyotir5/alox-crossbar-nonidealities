"""
AIHWKit baseline (run inside the aihwkit venv).
Trains a small MLP on MNIST in floating point, evaluates (1) software accuracy and
(2) AIHWKit analog in-memory inference (standard InferenceRPUConfig: programming
noise, read noise, output ADC), and saves the trained weights + a test subset so
our own calibrated simulator can be evaluated on the SAME network.

Output: mnist_mlp.npz (W1,b1,W2,b2,Xtest,ytest) + printed accuracies.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

torch.manual_seed(0); np.random.seed(0)

# ---- data ----
print("fetching MNIST ...", flush=True)
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
X = (mnist.data / 255.0).astype("float32")
y = mnist.target.astype(int)
# resize 28x28 -> 8x8 (bilinear), matching Keshari et al.'s MNIST setup (64 inputs)
import torch.nn.functional as F
X = F.interpolate(torch.tensor(X).reshape(-1, 1, 28, 28), size=(8, 8),
                  mode="bilinear", align_corners=False).reshape(-1, 64).numpy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=10000, random_state=0)
Xte, yte = Xte[:2000], yte[:2000]

# ---- float MLP 64-256-10 (8x8 MNIST, matching the reference device study) ----
H = 256
net = nn.Sequential(nn.Linear(64, H), nn.ReLU(), nn.Linear(H, 10))
opt = torch.optim.Adam(net.parameters(), 1e-3)
lossf = nn.CrossEntropyLoss()
Xtr_t, ytr_t = torch.tensor(Xtr), torch.tensor(ytr)
for ep in range(25):
    perm = torch.randperm(len(Xtr_t))
    for i in range(0, len(Xtr_t), 128):
        b = perm[i:i + 128]
        opt.zero_grad(); loss = lossf(net(Xtr_t[b]), ytr_t[b]); loss.backward(); opt.step()
Xte_t, yte_t = torch.tensor(Xte), torch.tensor(yte)
with torch.no_grad():
    acc_sw = (net(Xte_t).argmax(1) == yte_t).float().mean().item()
print(f"software MLP accuracy: {acc_sw*100:.2f}%", flush=True)

# save weights + test set for our simulator
sd = net.state_dict()
np.savez("mnist_mlp.npz",
         W1=sd["0.weight"].numpy(), b1=sd["0.bias"].numpy(),
         W2=sd["2.weight"].numpy(), b2=sd["2.bias"].numpy(),
         Xtest=Xte, ytest=yte, acc_sw=acc_sw)

# ---- AIHWKit analog inference baseline ----
try:
    from aihwkit.nn import AnalogLinear, AnalogSequential
    from aihwkit.simulator.configs import InferenceRPUConfig
    from aihwkit.simulator.configs.utils import WeightNoiseType

    rpu = InferenceRPUConfig()                  # standard PCM-like inference model
    amodel = AnalogSequential(
        AnalogLinear(64, H, bias=True, rpu_config=rpu), nn.ReLU(),
        AnalogLinear(H, 10, bias=True, rpu_config=rpu))
    # load the float weights into the analog layers
    al = [m for m in amodel.modules() if isinstance(m, AnalogLinear)]
    al[0].set_weights(torch.tensor(sd["0.weight"]), torch.tensor(sd["0.bias"]))
    al[1].set_weights(torch.tensor(sd["2.weight"]), torch.tensor(sd["2.bias"]))
    amodel.eval()
    amodel.program_analog_weights()
    accs = []
    for _ in range(5):                          # several noisy programmings
        with torch.no_grad():
            accs.append((amodel(Xte_t).argmax(1) == yte_t).float().mean().item())
    print(f"AIHWKit analog inference accuracy: {np.mean(accs)*100:.2f} +/- {np.std(accs)*100:.2f}%", flush=True)
except Exception as e:
    print("AIHWKit eval failed:", repr(e), flush=True)
