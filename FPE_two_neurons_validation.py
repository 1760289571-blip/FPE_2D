"""Validation simulation for two neurons with shared noise.

Ported from `FPE_two_neurons_validation.m` and extended to save probability
density data for quantitative comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

def run_simulation(
    tau_m: float,
    v_th: float,
    v_r: float,
    tau_r: float,
    sigma: float,
    i_0: float,
    c: float,
    t_dur: float,
    dt: float,
    n_neurons: int,
    burn_in_steps: int,
    bins: int,
    value_range: tuple[float, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate trajectories and estimate 2D stationary density with histogram."""
    rng = np.random.default_rng(seed)
    n_steps = int(np.floor(t_dur / dt))

    t_ref = np.zeros((n_steps + 1, n_neurons), dtype=np.float64)
    v = np.zeros((n_steps + 1, n_neurons), dtype=np.float64)
    v[0, :] = v_th * rng.random(n_neurons)

    for i in range(n_steps):
        if i%1000000:
            print('simulation process: time={}s'.format(i*dt))
        # refractory indicators at current step
        ind_ref = t_ref[i, :] > 0

        # decrease refractory timer for next step
        t_ref[i + 1, ind_ref] = t_ref[i, ind_ref] - dt

        # recover from refractory: set to reset voltage
        recover_now = ind_ref & (t_ref[i, :] <= 0)
        v[i, recover_now] = v_r

        # noise construction: shared + private
        priv_noise = rng.standard_normal(n_neurons)
        public_noise = rng.standard_normal()
        noise = np.sqrt(1.0 - c) * priv_noise + np.sqrt(c) * public_noise

        # dynamics dv = dt/tau_m * (-v + I0 + sigma * noise/sqrt(dt))
        dv = (dt / tau_m) * (-v[i, :] + i_0 + sigma * noise / np.sqrt(dt))
        active = ~ind_ref
        v[i + 1, active] = v[i, active] + dv[active]
        
        #fire and reset:
        ind_fire = v[i + 1, :] > v_th
        v[i + 1, ind_fire]=v_r
        t_ref[i + 1, ind_fire] = tau_r
        

    valid = (t_ref[:, 0] <= 0) & (t_ref[:, 1] <= 0)
    data = v[valid, :]
    if burn_in_steps > 0 and burn_in_steps < data.shape[0]:
        data = data[burn_in_steps:, :]

    hist, x_edges, y_edges = np.histogram2d(
        data[:, 0],
        data[:, 1],
        bins=bins,
        range=[[value_range[0], value_range[1]], [value_range[0], value_range[1]]],
        density=True,
    )

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return hist, x_centers, y_centers


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-neuron validation simulator")
    parser.add_argument("--tau_m", type=float, default=0.02)
    parser.add_argument("--v_th", type=float, default=20.0)
    parser.add_argument("--v_r", type=float, default=0.0)
    parser.add_argument("--tau_r", type=float, default=0.002)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--i_0", type=float, default=8.0)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--t_dur", type=float, default=1000.0)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--n_neurons", type=int, default=2)
    parser.add_argument("--burn_in_steps", type=int, default=1000000)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--range_min", type=float, default=-20.0)
    parser.add_argument("--range_max", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="validation_density.npz",
        help="Output .npz path for validation density",
    )
    args = parser.parse_args()

    density, x_centers, y_centers = run_simulation(
        tau_m=args.tau_m,
        v_th=args.v_th,
        v_r=args.v_r,
        tau_r=args.tau_r,
        sigma=args.sigma,
        i_0=args.i_0,
        c=args.c,
        t_dur=args.t_dur,
        dt=args.dt,
        n_neurons=args.n_neurons,
        burn_in_steps=args.burn_in_steps,
        bins=args.bins,
        value_range=(args.range_min, args.range_max),
        seed=args.seed,
    )
    
    # 画热力图
    plt.subplots(2, 3, figsize=(18, 12))
    plt.imshow(density.T,extent=[x_centers.min(),x_centers.max(),y_centers.min(),y_centers.max()],
               origin='lower', cmap='hot', aspect='auto')
    plt.savefig('Distribution_validation')
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        density=density,
        x=x_centers,
        y=y_centers,
        meta={
            "tau_m": args.tau_m,
            "v_th": args.v_th,
            "v_r": args.v_r,
            "tau_r": args.tau_r,
            "sigma": args.sigma,
            "i_0": args.i_0,
            "c": args.c,
            "t_dur": args.t_dur,
            "dt": args.dt,
            "bins": args.bins,
            "range": [args.range_min, args.range_max],
            "seed": args.seed,
        },
    )
    print(f"Validation density saved to: {out_path}")


if __name__ == "__main__":
    main()