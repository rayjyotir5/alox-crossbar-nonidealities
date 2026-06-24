"""Run a slice of the robotics operating-envelope grid and append results to JSONL.
Usage: python3 sweep.py --start S --count C --out results.jsonl"""
import argparse, json, itertools
from system_model import run_config

GRID = []
# main envelope grid: tile size x wire resistance x cell type (realistic ADC/DAC/noise)
for N in (32, 64, 96, 128):
    for rw in (0.5, 1.0, 2.0, 4.0):
        for cell in ("1R", "1T1R"):
            GRID.append(dict(N=N, r_w=rw, cell=cell, adc_bits=8, dac_bits=6,
                             read_noise=0.02, ideal=False, kind="grid"))
# ADC/DAC precision ablation at a fixed mid array
for ab, db in ((4, 4), (6, 6), (8, 8), (None, None)):
    GRID.append(dict(N=96, r_w=2.0, cell="1R", adc_bits=ab, dac_bits=db,
                     read_noise=0.02, ideal=False, kind="adcdac"))
# ideal-VMM reference (no IR, no quant) for the same tile sizes
for N in (32, 64, 96, 128):
    GRID.append(dict(N=N, r_w=2.0, cell="1R", adc_bits=None, dac_bits=None,
                     read_noise=0.0, ideal=True, kind="ideal"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=len(GRID))
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--n_test", type=int, default=150)
    a = ap.parse_args()
    with open(a.out, "a") as f:
        for cfg in GRID[a.start:a.start + a.count]:
            r = run_config({k: cfg[k] for k in ("N", "r_w", "cell", "adc_bits",
                            "dac_bits", "read_noise", "ideal")}, n_test=a.n_test)
            r["kind"] = cfg["kind"]
            f.write(json.dumps(r) + "\n"); f.flush()
            print(f"done {cfg['kind']} N={cfg['N']} rw={cfg['r_w']} {cfg['cell']} "
                  f"adc={cfg['adc_bits']} -> acc={r['acc']:.1f}% mvm={r['mvm_err']:.1f}%", flush=True)
    print("GRID_TOTAL", len(GRID))
