"""Experiment runner for the blur reconstruction benchmark."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dolfinx import fem
from matplotlib import cm
from scipy.interpolate import griddata

import bvp_blur
import opt_blur
import synthetic_problem
import trust_region_solver
import tv_coarse_handler


def plot_control(bvp, w_vec, *, title="control", iteration=0, file_prefix="result"):
    """Save a 2D contour plot for a DG0 control vector."""
    fig = plt.figure(figsize=(5, 5))
    z_grid = griddata(
        (bvp.X_gridpts_ctrl_plt, bvp.Y_gridpts_ctrl_plt),
        w_vec,
        (bvp.xgrid_ctrl_plt, bvp.ygrid_ctrl_plt),
        method="nearest",
    )
    ax = fig.add_subplot(1, 1, 1)
    ax.contourf(
        bvp.xgrid_ctrl_plt,
        bvp.ygrid_ctrl_plt,
        z_grid,
        cmap=cm.binary,
        antialiased=False,
    )
    ax.set_title(title)
    fig.tight_layout(pad=5.0)
    plt.savefig(f"{file_prefix}_plt_{iteration:01d}_w_ex.png", dpi=800)
    plt.close(fig)


def run_experiment():
    """Run one synthetic inverse problem instance and save visualizations."""
    start = time.time()
    nxny_bvp = 16
    alpha = 1e-3
    nu = 1e-2
    results_dir = Path("results") / f"n{nxny_bvp}_a{alpha:.0e}_nu{nu:.0e}"
    results_dir.mkdir(parents=True, exist_ok=True)

    f_exact_lbd, u_d_exact_lbd, _, _, w_exact_lbd = (
        synthetic_problem.build_synthetic_problem(alpha, nu)
    )

    bvp = bvp_blur.BvpBlur(nxny_bvp, nu, f_exact_lbd)
    opt = opt_blur.OptBlur(bvp, alpha, u_d_exact_lbd)

    delta0 = 0.125
    v0 = opt.get_zero_as_nparray().copy()

    w_fun = fem.Function(bvp.DG0)
    w_fun.interpolate(lambda x: w_exact_lbd(x[0], x[1]))
    w_exact_vec = w_fun.x.array
    u_sol_vec, _, _, _ = bvp.assemble_matrices_pde_adj(w_exact_vec)

    sigma = 1e-4
    c = np.sqrt(2.0)
    tv_helper = tv_coarse_handler.CoarseTVHandler(bvp, alpha)
    log_path = results_dir / "trust_region.log"
    w_sol = trust_region_solver.solve_trust_region_lazy(
        v0,
        delta0,
        sigma,
        alpha,
        c,
        bvp,
        opt,
        tv_helper,
        opt.tri_sqr.T @ w_exact_vec,
        output_file=str(log_path),
    )

    bvp.plot_u_3d(
        u_sol_vec, title="state", iter=0, fn_prefix=str(results_dir / "state")
    )
    bvp.plot_w(
        w_exact_vec,
        title="exact control",
        iter=0,
        fn_prefix=str(results_dir / "control_exact"),
    )
    plot_control(
        bvp,
        opt.tri_sqr @ w_sol,
        title="Recovered control",
        file_prefix=str(results_dir / "control_recovered"),
    )
    print(f"Elapsed time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    run_experiment()
