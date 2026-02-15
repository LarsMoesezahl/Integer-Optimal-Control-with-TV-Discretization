"""Coordinator for the trust-region TV subproblems."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mip_subproblem import solve_mip_subproblem


def solve_trust_region_subproblem(
    current_iterate: np.ndarray,
    gradient: np.ndarray,
    delta: float,
    alpha: float,
    c: float,
    d_csr,
    b_csr,
    hx: float,
    with_bnd: bool,
    optimizer: Any,
    h_tv: float,
    nsqr: int,
    dg0,
    rt0,
    m_matrix,
    bnd_dofs,
    dof_edge_ctrs,
    nrt0: int,
    base_coarse,
    tv_tilde: float,
):
    """Solve the nested trust-region subproblem.

    The method repeatedly solves sub-problem 1 and may append additional
    constraints in follow-up iterations. The current implementation keeps the
    loop limit and convergence heuristic from the original code.
    """
    start_time = time.time()
    count = 1
    rt_div_vector_list = [np.ones(nsqr) * 1e-3]
    v_value = -1.0
    candidate = np.zeros(optimizer.nsqr)
    tv_from_opt = 0.0
    tv_matrix = 0.0

    while tv_from_opt - v_value > 0.01 and count < 5:
        candidate, v_value, tv_matrix = solve_mip_subproblem(
            current_iterate,
            gradient,
            delta,
            alpha,
            c,
            d_csr,
            b_csr,
            hx,
            with_bnd,
            optimizer,
            count,
            rt_div_vector_list,
            base_coarse,
            nsqr,
            h_tv,
            dg0,
            rt0,
            m_matrix,
            bnd_dofs,
            dof_edge_ctrs,
            nrt0,
            tv_tilde,
        )
        count += 1

    elapsed = time.time() - start_time
    return candidate, count - 1, elapsed, v_value, tv_matrix


# Backward compatibility for previous API name.
sub_problem_cb = solve_trust_region_subproblem
