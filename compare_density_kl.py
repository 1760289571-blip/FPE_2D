"""Load saved densities and compare them quantitatively with KL divergence."""

from __future__ import annotations

import argparse
import numpy as np


def normalize_density(density: np.ndarray, x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    density = np.clip(density, eps, None)
    z = np.sum(density) * dx * dy
    return density / z


def kl_divergence(p: np.ndarray, q: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Continuous-grid approximation: KL(P||Q) = ∫ p log(p/q) dxdy."""
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return float(np.sum(p * np.log(p / q)) * dx * dy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two 2D densities with KL divergence")
    parser.add_argument("--validation", type=str, required=True, help="Path to validation_density.npz")
    parser.add_argument("--nn", type=str, required=True, help="Path to nn_density.npz")
    parser.add_argument("--eps", type=float, default=1e-12)
    args = parser.parse_args()

    val_data = np.load(args.validation, allow_pickle=True)
    nn_data = np.load(args.nn, allow_pickle=True)

    p_raw = val_data["density"]
    q_raw = nn_data["density"]
    x_p, y_p = val_data["x"], val_data["y"]
    x_q, y_q = nn_data["x"], nn_data["y"]

    if p_raw.shape != q_raw.shape or not np.allclose(x_p, x_q) or not np.allclose(y_p, y_q):
        print("validation shape={},nn_shape={}".format(p_raw.shape,q_raw.shape))
        raise ValueError(
            "Grid mismatch: make sure validation and NN density are saved on the same grid. "
            "Please set matching bins/range/grid_size."
        )

    p = normalize_density(p_raw, x_p, y_p, eps=args.eps)
    q = normalize_density(q_raw, x_q, y_q, eps=args.eps)

    kl_pq = kl_divergence(p, q, x_p, y_p)
    kl_qp = kl_divergence(q, p, x_p, y_p)
    js = 0.5 * kl_divergence(p, 0.5 * (p + q), x_p, y_p) + 0.5 * kl_divergence(q, 0.5 * (p + q), x_p, y_p)

    print(f"KL(validation || nn) = {kl_pq:.8e}")
    print(f"KL(nn || validation) = {kl_qp:.8e}")
    print(f"JS divergence         = {js:.8e}")


if __name__ == "__main__":
    main()