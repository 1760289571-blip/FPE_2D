#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation for Section 2.2 (steady-state firing rate) and data export for Section 2.3.

This script simulates the 2D Langevin model (v, g), estimates
1) empirical steady-state firing rate r0 from spike counts,
2) empirical stationary density,
and compares r0 with theoretical r0 exported by Tensor_2D_neuron_v_g.py.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def simulate_2d(params):
    rng = np.random.default_rng(params.seed)

    steps = int(params.t_max / params.dt)
    burn_steps = int(params.t_burn / params.dt)

    v = np.full(params.n_traj, params.v_init, dtype=np.float64)
    g = np.full(params.n_traj, params.g_init, dtype=np.float64)

    sqrt_dt = np.sqrt(params.dt)
    sigma_v = (params.sigma_ex / params.tau_m) * sqrt_dt
    sigma_g = (params.sigma_ffd * params.s_ee * np.sqrt(params.p_ee) / params.tau_ee) * sqrt_dt

    collected = []
    sample_gap = max(1, int(params.sample_every / params.dt))

    spike_count = 0
    measured_steps = 0

    for k in range(steps):
        dv_noise = sigma_v * rng.standard_normal(params.n_traj)
        dg_noise = sigma_g * rng.standard_normal(params.n_traj)

        v += ((-v + params.mu_ex + g) / params.tau_m) * params.dt + dv_noise
        g += ((-g + params.s_ee * params.mu_ffd * params.p_ee) / params.tau_ee) * params.dt + dg_noise

        spiked = v >= params.v_thres
        if k >= burn_steps:
            spike_count += int(np.count_nonzero(spiked))
            measured_steps += 1

        v[spiked] = params.v_reset

        if k >= burn_steps and (k - burn_steps) % sample_gap == 0:
            collected.append(np.stack((v.copy(), g.copy()), axis=1))

        if k % 10000 == 0:
            print(f"complete {k * params.dt / params.t_max * 100:.2f}%")

    if not collected:
        raise RuntimeError("No samples collected. Increase t_max or decrease sample_every.")

    samples = np.concatenate(collected, axis=0)
    measured_time = measured_steps * params.dt
    empirical_r0 = spike_count / (params.n_traj * measured_time + 1e-16)
    print("empirical firing rate:{}".format(empirical_r0))
    return samples, empirical_r0


def save_2d_density(samples, params):
    ranges = [(params.density_v_min, params.density_v_max), (params.density_g_min, params.density_g_max)]
    bins = [params.density_bins] * 2
    hist, edges = np.histogramdd(samples, bins=bins, range=ranges, density=True)
    centers = [0.5 * (axis_edges[:-1] + axis_edges[1:]) for axis_edges in edges]

    out_path = Path(params.density_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        density=hist,
        v=centers[0],
        g=centers[1],
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
            "mode": "2d",
        },
    )
    print(f"Saved 2D density data to: {out_path}")
    return hist


def compute_divergence_against_reference(sim_density, ref_density, eps=1e-15):
    p = np.asarray(sim_density, dtype=np.float64)
    q = np.asarray(ref_density, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"Density shape mismatch: sim {p.shape} vs ref {q.shape}")

    p = np.maximum(p, 0.0)
    q = np.maximum(q, 0.0)
    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)

    kl_pq = np.sum(p * np.log(p / q))
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return kl_pq, js


def compare_densities(sim_density, params):
    if not params.reference_density:
        return None, None

    ref = np.load(params.reference_density, allow_pickle=True)
    if "density" not in ref:
        raise ValueError(f"Reference file {params.reference_density} does not contain `density`.")

    kl, js = compute_divergence_against_reference(sim_density, ref["density"])
    print("\n--- Density validation (simulation vs reference P0) ---")
    print(f"KL(sim || ref): {kl:.6e}")
    print(f"JS(sim, ref):   {js:.6e}")
    return kl, js


def compare_firing_rate(empirical_r0, params):
    if not params.theory_summary:
        return None

    data = np.load(params.theory_summary, allow_pickle=True)
    if "theoretical_r0" not in data:
        raise ValueError(f"File {params.theory_summary} does not contain `theoretical_r0`.")

    theory_r0 = float(data["theoretical_r0"])
    abs_err = abs(empirical_r0 - theory_r0)
    rel_err = abs_err / max(abs(theory_r0), 1e-14)

    print("\n--- Section 2.2 firing-rate validation ---")
    print(f"Empirical r0 (SDE):      {empirical_r0:.8e}")
    print(f"Theoretical r0 (TNN P0): {theory_r0:.8e}")
    print(f"Absolute error:          {abs_err:.8e}")
    print(f"Relative error:          {rel_err:.8e}")

    return {
        "theory_r0": theory_r0,
        "abs_err": abs_err,
        "rel_err": rel_err,
    }


def save_validation_summary(params, empirical_r0, r0_stats, kl, js):
    out_path = Path(params.validation_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "empirical_r0": empirical_r0,
        "theory_r0": np.nan if r0_stats is None else r0_stats["theory_r0"],
        "r0_abs_err": np.nan if r0_stats is None else r0_stats["abs_err"],
        "r0_rel_err": np.nan if r0_stats is None else r0_stats["rel_err"],
        "density_kl": np.nan if kl is None else kl,
        "density_js": np.nan if js is None else js,
        "meta": {
            "dt": params.dt,
            "t_max": params.t_max,
            "t_burn": params.t_burn,
            "sample_every": params.sample_every,
            "n_traj": params.n_traj,
            "seed": params.seed,
            "mu_ex": params.mu_ex,
            "mu_ffd": params.mu_ffd,
            "tau_m": params.tau_m,
            "tau_ee": params.tau_ee,
            "sigma_ex": params.sigma_ex,
            "sigma_ffd": params.sigma_ffd,
            "p_ee": params.p_ee,
            "s_ee": params.s_ee,
        },
    }
    np.savez_compressed(out_path, **payload)
    print(f"Saved validation summary to: {out_path}")


def make_plots_2d(samples, params):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(samples[:, 0], bins=params.bins, density=True, alpha=0.8, color='tab:blue')
    axes[0].set_title('Marginal p(v)')
    axes[0].set_xlabel('v')

    axes[1].hist(samples[:, 1], bins=params.bins, density=True, alpha=0.8, color='tab:orange')
    axes[1].set_title('Marginal p(g)')
    axes[1].set_xlabel('g')

    h = axes[2].hist2d(
        samples[:, 0],
        samples[:, 1],
        bins=params.bins_2d,
        density=True,
        range=[[params.v_min, params.v_max], [params.g_min, params.g_max]],
        cmap='hot',
    )
    axes[2].set_title('Joint p(v, g)')
    axes[2].set_xlabel('v')
    axes[2].set_ylabel('g')
    fig.colorbar(h[3], ax=axes[2], label='density')

    fig.suptitle(
        f"2D SDE validation: mu_ex={params.mu_ex}, sigma_ex={params.sigma_ex}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(params.output, dpi=180)
    print(f"Saved figure to: {params.output}")


def build_argparser():
    p = argparse.ArgumentParser(description='2D SDE simulation for Section 2.2 validation')
    p.add_argument('--dt', type=float, default=1e-4)
    p.add_argument('--t_max', type=float, default=300.0)
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
    p.add_argument('--p_ee', type=float, default=0.2)
    p.add_argument('--s_ee', type=float, default=1.0)
    p.add_argument('--v_thres', type=float, default=20.0)
    p.add_argument('--v_reset', type=float, default=0.0)
    p.add_argument('--v_init', type=float, default=0.0)
    p.add_argument('--g_init', type=float, default=0.0)

    p.add_argument('--bins', type=int, default=120)
    p.add_argument('--bins_2d', type=int, default=100)
    p.add_argument('--output', type=str, default='sde_2d_density.png')
    p.add_argument('--v_min', type=float, default=-20.0)
    p.add_argument('--v_max', type=float, default=20.0)
    p.add_argument('--g_min', type=float, default=-10.0)
    p.add_argument('--g_max', type=float, default=10.0)

    p.add_argument('--density_bins', type=int, default=200)
    p.add_argument('--density_v_min', type=float, default=-20.0)
    p.add_argument('--density_v_max', type=float, default=20.0)
    p.add_argument('--density_g_min', type=float, default=-10.0)
    p.add_argument('--density_g_max', type=float, default=10.0)
    p.add_argument('--density_output', type=str, default='validation_2d_density.npz')

    p.add_argument('--reference_density', type=str, default='')
    p.add_argument('--theory_summary', type=str, default='')
    p.add_argument('--validation_output', type=str, default='validation_section2_2_summary.npz')

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    samples, empirical_r0 = simulate_2d(args)
    make_plots_2d(samples, args)

    sim_density = save_2d_density(samples, args)
    kl, js = compare_densities(sim_density, args)
    r0_stats = compare_firing_rate(empirical_r0, args)

    save_validation_summary(args, empirical_r0, r0_stats, kl, js)


if __name__ == '__main__':
    main()
