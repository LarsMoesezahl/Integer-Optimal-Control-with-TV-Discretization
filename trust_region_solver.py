"""Trust-region outer loop with callback-based TV subproblem solves."""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
from dolfinx import fem

from trust_region_subproblem import solve_trust_region_subproblem


def _format_iteration_line(
    outer_iter: int,
    inner_iter: int,
    objective: float,
    grad_norm: float,
    radius: float,
    predicted_red: float,
    actual_red: float,
    sub_iter: int,
    sub_time: float,
    tv_opt_value: float,
    tv_matrix_value: float,
    objective_wo_tv: float,
    l1_error: float,
    l2_error: float,
    rel_l1_error: float,
    rel_l2_error: float,
) -> str:
    """Build a fixed-width log line for one trust-region iteration."""
    return (
        "{:^10d} | {:^10d} | {:^10.11f} | {:^10.11f} | {:^10.4f} | "
        "{:^10.7f} | {:^10.7f} | {:^10d} | {:^10.8s} | {:^10.11f} | "
        "{:^10.11f} | {:^10.11f} | {:^10.5f} | {:^10.7f} | {:^10.5f} | "
        "{:^10.7f} |"
    ).format(
        outer_iter,
        inner_iter,
        objective,
        grad_norm,
        radius,
        predicted_red,
        actual_red,
        sub_iter,
        time.strftime("%H:%M:%S", time.gmtime(sub_time)),
        tv_opt_value,
        tv_matrix_value,
        objective_wo_tv,
        l1_error,
        l2_error,
        rel_l1_error,
        rel_l2_error,
    )


def solve_trust_region_lazy(
    v0: np.ndarray,
    delta0: float,
    sigma: float,
    alpha: float,
    c: float,
    bvp: Any,
    optimizer: Any,
    tv_helper: Any,
    w_exact: np.ndarray,
    max_outer_iter: int = 100,
    output_file: str = "output.txt",
) -> np.ndarray:
    """Solve the TV-regularized optimization problem via trust-region steps.

    Parameters
    ----------
    v0:
        Initial iterate on the optimization grid.
    delta0:
        Initial trust-region radius.
    sigma:
        Sufficient decrease parameter in the trust-region acceptance test.
    alpha:
        TV regularization weight.
    c:
        Scalar for the relation between primal TV surrogate and dual quantity.
    bvp, optimizer, tv_helper:
        Problem objects for PDE solves, objective/gradient evaluations,
        and coarse-grid TV structures.
    w_exact:
        Reference control for diagnostic error logging.
    max_outer_iter:
        Maximum number of outer trust-region iterations.
    output_file:
        Path to the log file.

    Returns
    -------
    numpy.ndarray
        Final accepted iterate.
    """
    test_start_time = time.time()

    with open(output_file, "w", encoding="utf-8") as log_file:
        log_file.write(
            f"Callback Trust Region Method with TV Regularization on a mesh "
            f"with {bvp.nx}x{bvp.ny}\n"
        )
        log_file.write(f"alpha={alpha}\n")
        log_file.write(f"sigma={sigma}\n")
        log_file.write(f"Delta0={delta0}\n\n\n")

        current_iterate = copy.deepcopy(v0)
        dg0 = fem.functionspace(optimizer.mesh, ("DG", 0))
        diff_fun = fem.Function(dg0)
        grad_fun = fem.Function(dg0)

        header = (
            "{:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | "
            "{:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10} | "
            "{:^10} | {:^10} | {:^10} |"
        ).format(
            "Iter n",
            "Iter k",
            "Objective",
            "Grad norm",
            "Radius",
            "pred",
            "ared",
            "Iter sub",
            "time sub",
            "TV opt",
            "TV matrix",
            "f w/o TV",
            "1norm diff",
            "2norm diff",
            "1norm gap",
            "2norm gap",
        )
        log_file.write(header + "\n")
        log_file.write("-" * len(header) + "\n")

        current_tv = optimizer.reg(current_iterate)

        for outer_it in range(1, max_outer_iter + 1):
            inner_it = 0
            delta_nk = delta0
            ared = 0.0
            pred = np.inf
            line = ""

            while ared < sigma * pred:
                gradient = optimizer.df(current_iterate)
                candidate, iter_sub, time_sub, tv_opt, tv_mat = (
                    solve_trust_region_subproblem(
                        current_iterate,
                        gradient,
                        delta_nk,
                        alpha,
                        c,
                        optimizer.D_csr,
                        optimizer.B_csr,
                        optimizer.hx,
                        optimizer.with_bnd,
                        optimizer,
                        tv_helper.hx_coarse,
                        tv_helper.nsqr_coarse,
                        tv_helper.DG0_coarse,
                        tv_helper.RT0_coarse,
                        tv_helper.M,
                        tv_helper.bnd_dofs,
                        tv_helper.dof_edge_ctrs,
                        tv_helper.nRT0,
                        tv_helper.base_coarse,
                        current_tv,
                    )
                )

                diff_fun.x.array[:] = current_iterate - candidate
                grad_fun.x.array[:] = gradient

                inner_product = np.sum(grad_fun.x.array[:] * diff_fun.x.array[:])
                pred = inner_product + alpha * current_tv - alpha * tv_opt

                candidate_objective = optimizer.f(candidate)
                current_objective = optimizer.f(current_iterate)
                ared = (
                    current_objective
                    + alpha * current_tv
                    - candidate_objective
                    - alpha * tv_opt
                )

                l1_err = np.linalg.norm(w_exact - candidate, 1)
                l2_err = np.linalg.norm(w_exact - candidate, 2)
                line = _format_iteration_line(
                    outer_it - 1,
                    inner_it,
                    current_objective + alpha * tv_opt,
                    np.linalg.norm(gradient, 2),
                    delta_nk,
                    pred,
                    ared,
                    iter_sub,
                    time_sub,
                    tv_opt,
                    tv_mat,
                    current_objective,
                    l1_err,
                    l2_err,
                    l1_err / np.linalg.norm(w_exact, 1),
                    l2_err / np.linalg.norm(w_exact, 2),
                )

                if pred <= 0.0:
                    total_time = time.strftime(
                        "%H:%M:%S", time.gmtime(time.time() - test_start_time)
                    )
                    log_file.write(line + "\n")
                    log_file.write(
                        "Terminating: Predicted reduction non-positive. "
                        f"Counter = {inner_it}\n"
                    )
                    log_file.write(
                        "Time consumed for complete Optimization = "
                        f"{total_time}\n"
                    )
                    log_file.flush()
                    return current_iterate

                if ared < sigma * pred:
                    log_file.write(line + "\n")
                    log_file.flush()
                    inner_it += 1
                    delta_nk /= 2.0
                else:
                    log_file.write(line + "\n")
                    current_iterate = candidate
                    current_tv = tv_opt
                    break

        total_time = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - test_start_time)
        )
        log_file.write(line + "\n\n\n")
        log_file.write(
            f"Terminating: max outer iteration count reached: {max_outer_iter}\n"
        )
        log_file.write(f"Time consumed for complete Optimization = {total_time}\n")

    return current_iterate


# Backward compatibility for previous API name.
slip_lazy = solve_trust_region_lazy
