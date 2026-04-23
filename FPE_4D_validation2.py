# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 23:19:25 2026

@author: 幽灵
"""

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
from pathlib import Path

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
        g1 += ((-g1 + params.s_ee * params.mu_ffd * params.p_ee) / params.tau_ee) * params.dt + dg_noise_1
        g2 += ((-g2 + params.s_ee * params.mu_ffd * params.p_ee) / params.tau_ee) * params.dt + dg_noise_2

        # Instantaneous reset when threshold is crossed (tau_r = 0)
        v1[v1 >= params.v_thres] = params.v_reset
        v2[v2 >= params.v_thres] = params.v_reset
        
        if not k%10000:
            print('complete {}%'.format(k*params.dt/params.t_max*100))
        if k >= burn_steps and (k - burn_steps) % sample_gap == 0:
            collected.append(np.stack((v1.copy(), v2.copy(), g1.copy(), g2.copy()), axis=1))

    if not collected:
        raise RuntimeError("No samples collected. Increase t_max or decrease sample_every.")

    samples = np.concatenate(collected, axis=0)
    return samples


def make_plots(samples, params):
    g1_is_zero=abs(samples[:,2]) <= 0.1
    g2_is_zero=abs(samples[:,3]) <= 0.1
    v1_is_zero=abs(samples[:,0]) <= 1
    v2_is_zero=abs(samples[:,1]) <= 1
    v_is_zero=np.logical_and(v1_is_zero,v2_is_zero)
    g1 = samples[v_is_zero, 2]
    g2 = samples[v_is_zero, 3]
    '''
    g1 = samples[:,2]
    g2 = samples[:,3]
    '''
    

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(g1, bins=params.bins, density=True, alpha=0.8, color='tab:blue')
    axes[0].set_title('Marginal p(g1)')
    axes[0].set_xlabel('g1')

    axes[1].hist(g2, bins=params.bins, density=True, alpha=0.8, color='tab:orange')
    axes[1].set_title('Marginal p(g2)')
    axes[1].set_xlabel('g2')

    h = axes[2].hist2d(g1, g2, bins=params.bins_2d, density=True, range=[[params.r_min,params.r_max],[params.r_min,params.r_max]],cmap='hot')
    axes[2].set_title('Joint p(g1, g2)')
    axes[2].set_xlabel('g1')
    axes[2].set_ylabel('g2')
    fig.colorbar(h[3], ax=axes[2], label='density')

    fig.suptitle(
        f"SDE validation (tau_r=0): mu_ex={params.mu_ex}, sigma_ex={params.sigma_ex}, c={params.c}",
        fontsize=12,
    )
    fig.tight_layout()

    out = params.output
    fig.savefig(out, dpi=180)
    print(f"Saved figure to: {out}")

def _safe_grad(arr, step, axis):
    return np.gradient(arr, step, axis=axis, edge_order=2)


def _safe_hess(arr, step, axis):
    first = _safe_grad(arr, step, axis)
    return _safe_grad(first, step, axis)


def _safe_cross(arr, step_a, step_b, axis_a, axis_b):
    d_a = _safe_grad(arr, step_a, axis_a)
    return _safe_grad(d_a, step_b, axis_b)


def check_steady_state_operator(samples, params):
    """Estimate P from simulation and evaluate (L+R)P on grid points.

    L is constructed from the Langevin drift/diffusion.
    R is approximated for tau_r=0 by reinjecting threshold flux to reset planes.
    """
    ranges = [
        (params.v_min, params.v_max),
        (params.v_min, params.v_max),
        (params.g_min, params.g_max),
        (params.g_min, params.g_max),
    ]
    bins = [params.grid_bins] * 4

    hist, edges = np.histogramdd(samples, bins=bins, range=ranges, density=False)
    dv1 = edges[0][1] - edges[0][0]
    dv2 = edges[1][1] - edges[1][0]
    dg1 = edges[2][1] - edges[2][0]
    dg2 = edges[3][1] - edges[3][0]
    cell_volume = dv1 * dv2 * dg1 * dg2

    P = hist / (np.sum(hist) * cell_volume + 1e-16)

    v1c = 0.5 * (edges[0][1:] + edges[0][:-1])
    v2c = 0.5 * (edges[1][1:] + edges[1][:-1])
    g1c = 0.5 * (edges[2][1:] + edges[2][:-1])
    g2c = 0.5 * (edges[3][1:] + edges[3][:-1])
    V1, V2, G1, G2 = np.meshgrid(v1c, v2c, g1c, g2c, indexing='ij')

    A1 = (-V1 + params.mu_ex + G1) / params.tau_m
    A2 = (-V2 + params.mu_ex + G2) / params.tau_m
    A3 = (-G1 + params.s_ee * params.mu_ffd * params.p_ee) / params.tau_ee
    A4 = (-G2 + params.s_ee * params.mu_ffd * params.p_ee) / params.tau_ee

    Dv = 0.5 * (params.sigma_ex / params.tau_m) ** 2
    Dg = 0.5 * (params.sigma_ffd * params.s_ee * np.sqrt(params.p_ee) / params.tau_ee) ** 2
    Dv12 = 0.5 * params.c * (params.sigma_ex / params.tau_m) ** 2
    Dg12 = 0.5 * params.p_ee * (params.sigma_ffd * params.s_ee * np.sqrt(params.p_ee) / params.tau_ee) ** 2

    LP = -_safe_grad(A1 * P, dv1, 0) - _safe_grad(A2 * P, dv2, 1) - _safe_grad(A3 * P, dg1, 2) - _safe_grad(A4 * P, dg2, 3)
    LP += Dv * _safe_hess(P, dv1, 0) + Dv * _safe_hess(P, dv2, 1)
    LP += Dg * _safe_hess(P, dg1, 2) + Dg * _safe_hess(P, dg2, 3)
    LP += 2.0 * Dv12 * _safe_cross(P, dv1, dv2, 0, 1)
    LP += 2.0 * Dg12 * _safe_cross(P, dg1, dg2, 2, 3)

    RP = np.zeros_like(P)
    if params.approx_reset and params.tau_r == 0.0:
        i_th = np.searchsorted(v1c, params.v_thres)
        j_th = np.searchsorted(v2c, params.v_thres)
        i_r = np.argmin(np.abs(v1c - params.v_reset))
        j_r = np.argmin(np.abs(v2c - params.v_reset))
        i_th = np.clip(i_th, 1, len(v1c) - 2)
        j_th = np.clip(j_th, 1, len(v2c) - 2)

        dP_dv1 = _safe_grad(P, dv1, 0)
        dP_dv2 = _safe_grad(P, dv2, 1)

        flux1 = np.maximum(0.0,  - Dv * dP_dv1[i_th] )
        flux2 = np.maximum(0.0,  - Dv * dP_dv2[:, j_th])

        RP[i_r, :, :, :] += flux1 / max(dv1, 1e-12)
        RP[:, j_r, :, :] += flux2 / max(dv2, 1e-12)

    total = LP + RP

    boundary_mask = (
        (np.abs(V1 - params.v_thres) < params.exclude_width)
        | (np.abs(V2 - params.v_thres) < params.exclude_width)
        | (np.abs(V1 - params.v_reset) < params.exclude_width)
        | (np.abs(V2 - params.v_reset) < params.exclude_width)
    )
    interior = ~boundary_mask

    l1_abs = np.mean(np.abs(total[interior]))
    linf_abs = np.max(np.abs(total[interior]))
    pref = np.mean(np.abs(P[interior])) + 1e-14
    print("\n--- Steady-state check: (L+R)P ≈ 0 ---")
    print(f"Grid bins per dim: {params.grid_bins}")
    print(f"Interior points used: {interior.sum()} / {interior.size}")
    print(f"mean(|(L+R)P|): {l1_abs:.4e}")
    print(f"max(|(L+R)P|):  {linf_abs:.4e}")
    print(f"mean(|(L+R)P|) / mean(|P|): {l1_abs / pref:.4e}")

    if params.residual_slice_output:
        # g1,g2 near 0 slice for quick visualization
        kg1 = np.argmin(np.abs(g1c - 0.0))
        kg2 = np.argmin(np.abs(g2c - 0.0))
        res2d = total[:, :, kg1, kg2]
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.4))
        h = ax.imshow(
            res2d.T,
            origin='lower',
            extent=[v1c[0], v1c[-1], v2c[0], v2c[-1]],
            aspect='auto',
            cmap='coolwarm',
        )
        ax.set_title('(L+R)P slice at g1≈0, g2≈0')
        ax.set_xlabel('v1')
        ax.set_ylabel('v2')
        fig.colorbar(h, ax=ax, label='residual')
        fig.tight_layout()
        fig.savefig(params.residual_slice_output, dpi=180)
        print(f"Saved residual slice to: {params.residual_slice_output}")


def save_4d_density(samples, params):
    """Estimate and save stationary density on a 4D grid."""
    ranges = [
        (params.density_v_min, params.density_v_max),
        (params.density_v_min, params.density_v_max),
        (params.density_g_min, params.density_g_max),
        (params.density_g_min, params.density_g_max),
    ]
    bins = [params.density_bins] * 4

    hist, edges = np.histogramdd(samples, bins=bins, range=ranges, density=True)
    centers = [0.5 * (axis_edges[:-1] + axis_edges[1:]) for axis_edges in edges]

    out_path = Path(params.density_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        density=hist,
        v1=centers[0],
        v2=centers[1],
        g1=centers[2],
        g2=centers[3],
        edges=edges,
        meta={
            "density_bins": params.density_bins,
            "v_range": [params.density_v_min, params.density_v_max],
            "g_range": [params.density_g_min, params.density_g_max],
            "dt": params.dt,
            "t_max": params.t_max,
            "t_burn": params.t_burn,
            "sample_every": params.sample_every,
            "n_traj": params.n_traj,
            "seed": params.seed,
        },
    )
    print(f"Saved 4D density data to: {out_path}")
    
def build_argparser():
    p = argparse.ArgumentParser(description='Two-neuron SDE simulation for FPE validation')
    p.add_argument('--dt', type=float, default=1e-4)
    p.add_argument('--t_max', type=float, default=600.0)
    p.add_argument('--t_burn', type=float, default=1.0)
    p.add_argument('--sample_every', type=float, default=5e-3)
    p.add_argument('--n_traj', type=int, default=4000)
    p.add_argument('--seed', type=int, default=123)

    p.add_argument('--mu_ex', type=float, default=8)
    p.add_argument('--mu_ffd', type=float, default=0)
    p.add_argument('--tau_m', type=float, default=0.02)
    p.add_argument('--tau_ee', type=float, default=0.004)
    p.add_argument('--sigma_ex', type=float, default=2)
    p.add_argument('--sigma_ffd', type=float, default=0.8)
    p.add_argument('--c', type=float, default=0.5)
    p.add_argument('--p_ee', type=float, default=0.2)
    p.add_argument('--s_ee', type=float, default=1.0)
    p.add_argument('--v_thres', type=float, default=20.0)
    p.add_argument('--v_reset', type=float, default=0.0)
    p.add_argument('--tau_r', type=float, default=0.0)
    p.add_argument('--v_init', type=float, default=0.0)
    p.add_argument('--g_init', type=float, default=0.0)

    p.add_argument('--bins', type=int, default=120)
    p.add_argument('--bins_2d', type=int, default=100)
    p.add_argument('--output', type=str, default='sde_voltage_density.png')
    
    p.add_argument('--r_min', type=int, default=-10)
    p.add_argument('--r_max', type=int, default=10)
    p.add_argument('--check_steady', default=True, action='store_true')
    p.add_argument('--grid_bins', type=int, default=24)
    p.add_argument('--v_min', type=float, default=-20.0)
    p.add_argument('--v_max', type=float, default=20.0)
    p.add_argument('--g_min', type=float, default=-10.0)
    p.add_argument('--g_max', type=float, default=10.0)
    p.add_argument('--exclude_width', type=float, default=0.8)
    p.add_argument('--approx_reset', default=True, action='store_true')
    p.add_argument('--residual_slice_output', type=str, default='steady_residual_slice.png')

    p.add_argument('--density_bins', type=int, default=50)
    p.add_argument('--density_v_min', type=float, default=-20.0)
    p.add_argument('--density_v_max', type=float, default=20.0)
    p.add_argument('--density_g_min', type=float, default=-10.0)
    p.add_argument('--density_g_max', type=float, default=10.0)
    p.add_argument('--density_output', type=str, default='validation_4d_density.npz')

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    samples = simulate(args)
    make_plots(samples, args)
    save_4d_density(samples, args)
    if args.check_steady:
        check_steady_state_operator(samples, args)

if __name__ == '__main__':
    main()

