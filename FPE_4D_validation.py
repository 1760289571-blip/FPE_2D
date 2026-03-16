# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 21:01:09 2026

@author: 幽灵
"""

#!/usr/bin/env python3
"""Monte Carlo validation for the two-neuron conductance model.

This script simulates the 4D Langevin system used by Tensor_4D_neuron.py:
state = (v1, v2, g1, g2), with instantaneous reset (tau_r = 0).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def simulate(params):
    rng = np.random.default_rng(params.seed)

    steps = int(params.t_max / params.dt)
    burn_steps = int(params.t_burn / params.dt)

    # State arrays
    v1 = np.full(params.n_traj, params.v_init, dtype=np.float64)
    v2 = np.full(params.n_traj, params.v_init, dtype=np.float64)
    g1 = np.full(params.n_traj, params.g_init, dtype=np.float64)
    g2 = np.full(params.n_traj, params.g_init, dtype=np.float64)

    # Euler-Maruyama noise scales
    sqrt_dt = np.sqrt(params.dt)
    sigma_v = (params.sigma_ex / params.tau_m) * sqrt_dt
    sigma_g = (params.sigma_ffd * params.s_ee * np.sqrt(params.p_ee) / params.tau_ee) * sqrt_dt

    collected = []
    sample_gap = max(1, int(params.sample_every / params.dt))

    for k in range(steps):
        # Shared + private voltage noises
        xi0 = rng.standard_normal(params.n_traj)
        xi1 = rng.standard_normal(params.n_traj)
        xi2 = rng.standard_normal(params.n_traj)
        dv_noise_1 = sigma_v * (np.sqrt(1.0 - params.c) * xi1 + np.sqrt(params.c) * xi0)
        dv_noise_2 = sigma_v * (np.sqrt(1.0 - params.c) * xi2 + np.sqrt(params.c) * xi0)

        # Independent conductance noises
        dWg0 = rng.standard_normal(params.n_traj)
        dWg1 = rng.standard_normal(params.n_traj)
        dWg2 = rng.standard_normal(params.n_traj)

        dg_noise_1 = sigma_g * (np.sqrt(1.0 - params.p_ee) * dWg1 + np.sqrt(params.p_ee) * dWg0)
        dg_noise_2 = sigma_g * (np.sqrt(1.0 - params.p_ee) * dWg2 + np.sqrt(params.p_ee) * dWg0)

        v1 += ((-v1 + params.mu_ex + g1) / params.tau_m) * params.dt + dv_noise_1
        v2 += ((-v2 + params.mu_ex + g2) / params.tau_m) * params.dt + dv_noise_2
        g1 += ((-g1 + params.s_ee * params.mu_ffd * params.p_ee) / params.tau_ee) * params.dt + sigma_g * dg_noise_1
        g2 += ((-g2 + params.s_ee * params.mu_ffd * params.p_ee) / params.tau_ee) * params.dt + sigma_g * dg_noise_2

        # Instantaneous reset when threshold is crossed (tau_r = 0)
        v1[v1 >= params.v_thres] = params.v_reset
        v2[v2 >= params.v_thres] = params.v_reset

        if k >= burn_steps and (k - burn_steps) % sample_gap == 0:
            collected.append(np.stack((v1.copy(), v2.copy(), g1.copy(), g2.copy()), axis=1))

    if not collected:
        raise RuntimeError("No samples collected. Increase t_max or decrease sample_every.")

    samples = np.concatenate(collected, axis=0)
    return samples


def make_plots(samples, params):
    g1_is_zero=abs(samples[:,2]) <= 1
    g2_is_zero=abs(samples[:,3]) <= 1
    g_is_zero=np.logical_and(g1_is_zero,g2_is_zero)
    v1 = samples[g_is_zero, 0]
    v2 = samples[g_is_zero, 1]

    
    

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(v1, bins=params.bins, density=True, alpha=0.8, color='tab:blue')
    axes[0].set_title('Marginal p(g1)')
    axes[0].set_xlabel('g1')

    axes[1].hist(v2, bins=params.bins, density=True, alpha=0.8, color='tab:orange')
    axes[1].set_title('Marginal p(g2)')
    axes[1].set_xlabel('g2')

    h = axes[2].hist2d(v1, v2, bins=params.bins_2d, density=True, range=[[params.r_min,params.r_max],[params.r_min,params.r_max]],cmap='hot')
    axes[2].set_title('Joint p(v1, v2)')
    axes[2].set_xlabel('v1')
    axes[2].set_ylabel('v2')
    fig.colorbar(h[3], ax=axes[2], label='density')

    fig.suptitle(
        f"SDE validation (tau_r=0): mu_ex={params.mu_ex}, sigma_ex={params.sigma_ex}, c={params.c}",
        fontsize=12,
    )
    fig.tight_layout()

    out = params.output
    fig.savefig(out, dpi=180)
    print(f"Saved figure to: {out}")


def build_argparser():
    p = argparse.ArgumentParser(description='Two-neuron SDE simulation for FPE validation')
    p.add_argument('--dt', type=float, default=1e-4)
    p.add_argument('--t_max', type=float, default=6.0)
    p.add_argument('--t_burn', type=float, default=1.0)
    p.add_argument('--sample_every', type=float, default=5e-3)
    p.add_argument('--n_traj', type=int, default=4000)
    p.add_argument('--seed', type=int, default=123)

    p.add_argument('--mu_ex', type=float, default=8.0)
    p.add_argument('--mu_ffd', type=float, default=0)
    p.add_argument('--tau_m', type=float, default=0.02)
    p.add_argument('--tau_ee', type=float, default=0.004)
    p.add_argument('--sigma_ex', type=float, default=2.0)
    p.add_argument('--sigma_ffd', type=float, default=0.8)
    p.add_argument('--c', type=float, default=0.5)
    p.add_argument('--p_ee', type=float, default=0.2)
    p.add_argument('--s_ee', type=float, default=1.0)
    p.add_argument('--v_thres', type=float, default=20.0)
    p.add_argument('--v_reset', type=float, default=0.0)

    p.add_argument('--v_init', type=float, default=0.0)
    p.add_argument('--g_init', type=float, default=0.0)

    p.add_argument('--bins', type=int, default=120)
    p.add_argument('--bins_2d', type=int, default=100)
    p.add_argument('--output', type=str, default='sde_voltage_density.png')
    
    p.add_argument('--r_min', type=int, default=-20)
    p.add_argument('--r_max', type=int, default=20)
    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    samples = simulate(args)
    make_plots(samples, args)


if __name__ == '__main__':
    main()
