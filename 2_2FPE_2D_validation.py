#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation utilities for Section 2.2 and Section 2.3.

Section 2.2:
- Simulate the 2D SDE (v, g)
- Validate steady-state firing rate and density against TNN P0 output.

Section 2.3:
- Simulate spike events in stationarity
- Build event-triggered density P(v,g,t | spike at t=0)
- Compute Q(v,g,omega) = ∫ exp(-i*omega*t) * (P(v,g,t)-P0(v,g)) dt
- Compare simulation-based Q with TNN-based Q (U+iV) if provided.
- Save intermediates for Section 2.4.
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
    print(f"empirical firing rate: {empirical_r0}")
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


def _simulate_spike_triggered_density(params):
    """Estimate P(v,g,t | spike at t=0) and P0(v,g) from direct simulation."""
    rng = np.random.default_rng(params.seed + 100)

    dt = params.dt
    steps = int(params.sec23_t_total / dt)
    burn_steps = int(params.sec23_t_burn / dt)
    lag_steps = int(params.sec23_lag_tmax / dt)
    sample_lag_gap = max(1, int(params.sec23_lag_dt / dt))
    lag_indices = np.arange(0, lag_steps + 1, sample_lag_gap, dtype=np.int64)
    n_lags = len(lag_indices)

    v = np.full(params.sec23_n_traj, params.v_init, dtype=np.float64)
    g = np.full(params.sec23_n_traj, params.g_init, dtype=np.float64)

    sqrt_dt = np.sqrt(dt)
    sigma_v = (params.sigma_ex / params.tau_m) * sqrt_dt
    sigma_g = (params.sigma_ffd * params.s_ee * np.sqrt(params.p_ee) / params.tau_ee) * sqrt_dt

    # For P0 estimation
    p0_samples = []
    p0_gap = max(1, int(params.sec23_p0_every / dt))

    # Spike-event queue: each item is dict with reference trajectory index and remaining lag steps.
    active_events = []
    event_store = [[] for _ in range(n_lags)]
    max_events = params.sec23_max_events
    accepted_events = 0
    spike_count = 0
    measured_steps = 0

    for k in range(steps):
        dv_noise = sigma_v * rng.standard_normal(params.sec23_n_traj)
        dg_noise = sigma_g * rng.standard_normal(params.sec23_n_traj)

        v += ((-v + params.mu_ex + g) / params.tau_m) * dt + dv_noise
        g += ((-g + params.s_ee * params.mu_ffd * params.p_ee) / params.tau_ee) * dt + dg_noise

        spiked = v >= params.v_thres

        if k >= burn_steps:
            measured_steps += 1
            spike_count += int(np.count_nonzero(spiked))

            if (k - burn_steps) % p0_gap == 0:
                p0_samples.append(np.stack((v.copy(), g.copy()), axis=1))

            if accepted_events < max_events:
                idx = np.flatnonzero(spiked)
                if idx.size > 0:
                    left = max_events - accepted_events
                    picked = idx[:left]
                    for j in picked:
                        active_events.append({"traj": int(j), "start": k})
                        accepted_events += 1

        v[spiked] = params.v_reset

        # Collect event-triggered states
        if active_events:
            new_active = []
            for ev in active_events:
                lag = k - ev["start"]
                if lag in lag_indices:
                    lag_pos = int(np.where(lag_indices == lag)[0][0])
                    tr = ev["traj"]
                    event_store[lag_pos].append((v[tr], g[tr]))
                if lag < lag_steps:
                    new_active.append(ev)
            active_events = new_active

        if k % 10000 == 0:
            print(f"[sec2.3] complete {k * dt / max(params.sec23_t_total, 1e-12) * 100:.2f}%")

    if len(p0_samples) == 0:
        raise RuntimeError("No P0 samples collected for Section 2.3.")

    measured_time = measured_steps * dt
    empirical_r0 = spike_count / (params.sec23_n_traj * measured_time + 1e-16)

    p0_samples = np.concatenate(p0_samples, axis=0)

    ranges = [[params.sec23_v_min, params.sec23_v_max], [params.sec23_g_min, params.sec23_g_max]]
    bins = [params.sec23_bins, params.sec23_bins]

    p0_hist, v_edges, g_edges = np.histogram2d(
        p0_samples[:, 0],
        p0_samples[:, 1],
        bins=bins,
        range=ranges,
        density=True,
    )

    p_event = np.zeros((n_lags, params.sec23_bins, params.sec23_bins), dtype=np.float64)
    valid_counts = np.zeros(n_lags, dtype=np.int64)
    for i, lag_samples in enumerate(event_store):
        if len(lag_samples) == 0:
            continue
        arr = np.asarray(lag_samples, dtype=np.float64)
        valid_counts[i] = arr.shape[0]
        hist, _, _ = np.histogram2d(arr[:, 0], arr[:, 1], bins=bins, range=ranges, density=True)
        p_event[i] = hist

    v_centers = 0.5 * (v_edges[:-1] + v_edges[1:])
    g_centers = 0.5 * (g_edges[:-1] + g_edges[1:])

    return {
        "r0": float(empirical_r0),
        "p0": p0_hist,
        "p_event": p_event,
        "lag_times": lag_indices * dt,
        "lag_indices": lag_indices,
        "valid_counts": valid_counts,
        "v": v_centers,
        "g": g_centers,
    }


def compute_q_from_delta_p(p_event, p0, lag_times, omega):
    """Compute Q(v,g,omega) from ΔP(t)=P(t)-P0 using trapezoidal integration."""
    delta = p_event - p0[None, :, :]
    kernel = np.exp(-1j * omega * lag_times)[:, None, None]
    q_complex = np.trapezoid(kernel * delta, x=lag_times, axis=0)
    return q_complex.real, q_complex.imag


def _interp_q_to_grid(q_source, src_v, src_g, dst_v, dst_g):
    """Bilinear interpolation from source regular grid to destination regular grid."""
    tmp = np.empty((len(src_g), len(dst_v)), dtype=np.float64)
    for j in range(len(src_g)):
        tmp[j, :] = np.interp(dst_v, src_v, q_source[j, :], left=np.nan, right=np.nan)

    out = np.empty((len(dst_g), len(dst_v)), dtype=np.float64)
    for i in range(len(dst_v)):
        out[:, i] = np.interp(dst_g, src_g, tmp[:, i], left=np.nan, right=np.nan)
    return out


def validate_section_2_3(params):
    est = _simulate_spike_triggered_density(params)

    omega = params.sec23_omega
    q_real_mc, q_imag_mc = compute_q_from_delta_p(est["p_event"], est["p0"], est["lag_times"], omega)

    q_ref_real = None
    q_ref_imag = None
    q_real_rmse = np.nan
    q_imag_rmse = np.nan

    if params.sec23_reference_q:
        q_data = np.load(params.sec23_reference_q, allow_pickle=True)
        if not all(k in q_data for k in ("x", "g", "U", "V")):
            raise ValueError("Reference Q file must contain x, g, U, V.")

        src_v = np.asarray(q_data["x"])
        src_g = np.asarray(q_data["g"])
        q_ref_real = _interp_q_to_grid(np.asarray(q_data["U"]), src_v, src_g, est["v"], est["g"])
        q_ref_imag = _interp_q_to_grid(np.asarray(q_data["V"]), src_v, src_g, est["v"], est["g"])

        valid = np.isfinite(q_ref_real) & np.isfinite(q_ref_imag)
        q_real_rmse = float(np.sqrt(np.nanmean((q_real_mc[valid] - q_ref_real[valid]) ** 2)))
        q_imag_rmse = float(np.sqrt(np.nanmean((q_imag_mc[valid] - q_ref_imag[valid]) ** 2)))

        print("\n--- Section 2.3 Q-validation (MC vs TNN) ---")
        print(f"omega:         {omega:.6g}")
        print(f"RMSE Re(Q):    {q_real_rmse:.6e}")
        print(f"RMSE Im(Q):    {q_imag_rmse:.6e}")

    out_npz = Path(params.sec23_output)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        omega=omega,
        empirical_r0=est["r0"],
        v=est["v"],
        g=est["g"],
        p0=est["p0"],
        p_event=est["p_event"],
        lag_times=est["lag_times"],
        lag_indices=est["lag_indices"],
        valid_counts=est["valid_counts"],
        q_real_mc=q_real_mc,
        q_imag_mc=q_imag_mc,
        q_real_ref=np.array([]) if q_ref_real is None else q_ref_real,
        q_imag_ref=np.array([]) if q_ref_imag is None else q_ref_imag,
        q_real_rmse=q_real_rmse,
        q_imag_rmse=q_imag_rmse,
        meta={
            "dt": params.dt,
            "sec23_t_total": params.sec23_t_total,
            "sec23_t_burn": params.sec23_t_burn,
            "sec23_lag_tmax": params.sec23_lag_tmax,
            "sec23_lag_dt": params.sec23_lag_dt,
            "sec23_n_traj": params.sec23_n_traj,
            "sec23_max_events": params.sec23_max_events,
            "definition": "Q(v,g,omega)=int exp(-i*omega*t)*(P(v,g,t)-P0(v,g)) dt",
        },
    )
    print(f"Saved Section 2.3 data to: {out_npz}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    im0 = axes[0].imshow(
        q_real_mc,
        extent=[est["v"].min(), est["v"].max(), est["g"].min(), est["g"].max()],
        origin='lower',
        aspect='auto',
        cmap='coolwarm',
    )
    axes[0].set_title(f"MC Re(Q), ω={omega:.3g}")
    axes[0].set_xlabel("v")
    axes[0].set_ylabel("g")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(
        q_imag_mc,
        extent=[est["v"].min(), est["v"].max(), est["g"].min(), est["g"].max()],
        origin='lower',
        aspect='auto',
        cmap='coolwarm',
    )
    axes[1].set_title(f"MC Im(Q), ω={omega:.3g}")
    axes[1].set_xlabel("v")
    axes[1].set_ylabel("g")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(params.sec23_plot, dpi=180)
    plt.close(fig)
    print(f"Saved Section 2.3 Q plot to: {params.sec23_plot}")


def validate_section_2_4(params):
    """Validate theoretical S_self(omega) using direct spike-train simulation."""
    rng = np.random.default_rng(params.seed + 200)

    dt = params.dt
    steps = int(params.sec24_t_total / dt)
    burn_steps = int(params.sec24_t_burn / dt)
    omega = float(params.sec24_omega)

    v = np.full(params.sec24_n_traj, params.v_init, dtype=np.float64)
    g = np.full(params.sec24_n_traj, params.g_init, dtype=np.float64)

    sqrt_dt = np.sqrt(dt)
    sigma_v = (params.sigma_ex / params.tau_m) * sqrt_dt
    sigma_g = (params.sigma_ffd * params.s_ee * np.sqrt(params.p_ee) / params.tau_ee) * sqrt_dt

    # Fourier accumulator for each trajectory: sum_k exp(-i*omega*t_k)
    fourier_sum = np.zeros(params.sec24_n_traj, dtype=np.complex128)
    spike_count = np.zeros(params.sec24_n_traj, dtype=np.int64)

    for k in range(steps):
        t_now = k * dt

        dv_noise = sigma_v * rng.standard_normal(params.sec24_n_traj)
        dg_noise = sigma_g * rng.standard_normal(params.sec24_n_traj)

        v += ((-v + params.mu_ex + g) / params.tau_m) * dt + dv_noise
        g += ((-g + params.s_ee * params.mu_ffd * params.p_ee) / params.tau_ee) * dt + dg_noise

        spiked = v >= params.v_thres

        if k >= burn_steps and np.any(spiked):
            fourier_sum[spiked] += np.exp(-1j * omega * t_now)
            spike_count[spiked] += 1

        v[spiked] = params.v_reset

        if k % 10000 == 0:
            print(f"[sec2.4] complete {k * dt / max(params.sec24_t_total, 1e-12) * 100:.2f}%")

    t_meas = max((steps - burn_steps) * dt, 1e-12)
    s_emp_per_traj = (np.abs(fourier_sum) ** 2) / t_meas
    s_empirical = float(np.mean(s_emp_per_traj))
    r0_empirical = float(np.mean(spike_count) / t_meas)

    s_theory = np.nan
    r0_theory = np.nan
    if params.sec24_theory_psd:
        data = np.load(params.sec24_theory_psd, allow_pickle=True)
        if "s_self_omega" in data:
            s_theory = float(data["s_self_omega"])
        if "theoretical_r0" in data:
            r0_theory = float(data["theoretical_r0"])
        elif "r0" in data:
            r0_theory = float(data["r0"])

    abs_err = np.nan if not np.isfinite(s_theory) else abs(s_empirical - s_theory)
    rel_err = np.nan if not np.isfinite(s_theory) else abs_err / max(abs(s_theory), 1e-14)

    print("\n--- Section 2.4 PSD validation ---")
    print(f"omega:                    {omega:.6g}")
    print(f"Empirical r0 (SDE):       {r0_empirical:.8e}")
    if np.isfinite(r0_theory):
        print(f"Theoretical r0 (TNN):     {r0_theory:.8e}")
    print(f"Empirical S_self(omega):  {s_empirical:.8e}")
    if np.isfinite(s_theory):
        print(f"Theoretical S_self(omega):{s_theory:.8e}")
        print(f"Absolute error:           {abs_err:.8e}")
        print(f"Relative error:           {rel_err:.8e}")

    out_path = Path(params.sec24_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        omega=omega,
        s_empirical=s_empirical,
        s_empirical_per_traj=s_emp_per_traj,
        r0_empirical=r0_empirical,
        s_theory=s_theory,
        r0_theory=r0_theory,
        abs_err=abs_err,
        rel_err=rel_err,
        meta={
            "dt": dt,
            "t_total": params.sec24_t_total,
            "t_burn": params.sec24_t_burn,
            "n_traj": params.sec24_n_traj,
            "formula": "S_emp(omega)=mean_j(|sum_k exp(-i*omega*t_k^{(j)})|^2 / T)",
        },
    )
    print(f"Saved Section 2.4 validation summary to: {out_path}")


def build_argparser():
    p = argparse.ArgumentParser(description='2D SDE validation for Section 2.2, Section 2.3 and Section 2.4')
    p.add_argument('--mode', type=str, default='section2_2', choices=['section2_2', 'section2_3', 'section2_4'])

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

    # Section 2.3 validation settings
    p.add_argument('--sec23_omega', type=float, default=10.0)
    p.add_argument('--sec23_t_total', type=float, default=120.0)
    p.add_argument('--sec23_t_burn', type=float, default=5.0)
    p.add_argument('--sec23_lag_tmax', type=float, default=0.2)
    p.add_argument('--sec23_lag_dt', type=float, default=1e-3)
    p.add_argument('--sec23_n_traj', type=int, default=3000)
    p.add_argument('--sec23_max_events', type=int, default=50000)
    p.add_argument('--sec23_p0_every', type=float, default=5e-3)
    p.add_argument('--sec23_bins', type=int, default=120)
    p.add_argument('--sec23_v_min', type=float, default=-20.0)
    p.add_argument('--sec23_v_max', type=float, default=20.0)
    p.add_argument('--sec23_g_min', type=float, default=-10.0)
    p.add_argument('--sec23_g_max', type=float, default=10.0)
    p.add_argument('--sec23_reference_q', type=str, default='')
    p.add_argument('--sec23_output', type=str, default='validation_section2_3_Q_data.npz')
    p.add_argument('--sec23_plot', type=str, default='validation_section2_3_Q.png')

    # Section 2.4 validation settings
    p.add_argument('--sec24_omega', type=float, default=10.0)
    p.add_argument('--sec24_t_total', type=float, default=240.0)
    p.add_argument('--sec24_t_burn', type=float, default=10.0)
    p.add_argument('--sec24_n_traj', type=int, default=2000)
    p.add_argument('--sec24_theory_psd', type=str, default='')
    p.add_argument('--sec24_output', type=str, default='validation_section2_4_psd_summary.npz')

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.mode == 'section2_2':
        samples, empirical_r0 = simulate_2d(args)
        make_plots_2d(samples, args)

        sim_density = save_2d_density(samples, args)
        kl, js = compare_densities(sim_density, args)
        r0_stats = compare_firing_rate(empirical_r0, args)

        save_validation_summary(args, empirical_r0, r0_stats, kl, js)
    elif args.mode == 'section2_3':
        validate_section_2_3(args)
    else:
        validate_section_2_4(args)


if __name__ == '__main__':
    main()
