"""MIP solver for trust-region subproblem 1 with lazy TV constraints."""

from __future__ import annotations

from typing import Any

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from rt0_dual_solver import solve_rt0_divergence_subproblem


LAZY_TV_TOL = 1e-2


def lazy_callback(model: gp.Model, where: int) -> None:
    """Add violated TV cuts whenever an integer solution is found."""
    if where != GRB.Callback.MIPSOL:
        return

    u_val = model.cbGetSolution(model._u)
    div_phi, tv_from_opt = solve_rt0_divergence_subproblem(
        model._nsqr,
        model._h_tv,
        u_val,
        model._DG0,
        model._RT0,
        model._M,
        model._bnd_dofs,
        model._dof_edge_ctrs,
        model._nRT0,
        model._base_to_coarse,
    )
    v_val = model.cbGetSolution(model._V)

    if tv_from_opt - v_val > LAZY_TV_TOL:
        model.cbLazy(
            gp.quicksum(div_phi[i] * model._w_vec[i] for i in range(model._coarse_nsqr))
            <= model._V
        )


def solve_mip_subproblem(
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
    count: int,
    rt_div_vector_list: list[np.ndarray],
    base_coarse,
    coarse_nsqr: int,
    h_tv: float,
    dg0_coarse,
    rt0_coarse,
    m_coarse,
    bnd_dofs_coarse,
    dof_edge_ctrs_coarse,
    nrt0_coarse: int,
    tv_tilde: float,
):
    """Solve one trust-region MIP with TV surrogate variable `V`.

    Returns
    -------
    tuple
        `(u_opt, V_value, tv_interior_value)` when optimal.
    """
    n = optimizer.nsqr
    m = d_csr.shape[0]

    x_min, y_min = optimizer.pt_bottom_left
    x_max, y_max = optimizer.pt_top_right
    v = ((x_max - x_min) * (y_max - y_min)) / (optimizer.nx * optimizer.ny)

    if with_bnd:
        b = b_csr.shape[0]

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    model = gp.Model("TV_MIP", env=env)
    model.Params.LazyConstraints = 1

    u = model.addMVar(n, vtype=GRB.BINARY, name="u")
    t = model.addVars(range(n), vtype=GRB.CONTINUOUS, lb=0.0, name="t")
    z = model.addVars(range(m), vtype=GRB.CONTINUOUS, lb=0.0, name="z")
    v_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="V")

    u.start = current_iterate

    objective = gp.quicksum(gradient[i] * (u[i] - current_iterate[i]) for i in range(n))
    objective += alpha * v_var - alpha * tv_tilde
    model.setObjective(objective, GRB.MINIMIZE)

    for i in range(n):
        model.addConstr(t[i] >= (u[i] - current_iterate[i]) * v, name=f"abs1_{i}")
        model.addConstr(t[i] >= (current_iterate[i] - u[i]) * v, name=f"abs2_{i}")
    model.addConstr(gp.quicksum(t[i] for i in range(n)) <= delta, name="trust_region")

    d_data, d_indices, d_indptr = d_csr.data, d_csr.indices, d_csr.indptr
    for j in range(m):
        start, end = d_indptr[j], d_indptr[j + 1]
        indices = d_indices[start:end]
        coefs = d_data[start:end]
        tv_expr = gp.quicksum(coefs[k] * u[indices[k]] for k in range(len(indices)))
        model.addConstr(z[j] >= tv_expr, name=f"tv_int_pos_{j}")
        model.addConstr(z[j] >= -tv_expr, name=f"tv_int_neg_{j}")

    w_vec = model.addVars(
        range(coarse_nsqr),
        vtype=GRB.CONTINUOUS,
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        name="w_vec",
    )

    base_data = base_coarse.data
    base_indices = base_coarse.indices
    base_indptr = base_coarse.indptr
    n_coarse = base_coarse.shape[0]

    base_exprs = []
    for j in range(n_coarse):
        expr = gp.LinExpr()
        start, end = base_indptr[j], base_indptr[j + 1]
        for k in range(start, end):
            i = base_indices[k]
            expr += base_data[k] * u[i]
        expr += -1.0 * w_vec[j]
        base_exprs.append(expr)

    model.addConstrs(
        (base_exprs[j] == 0.0 for j in range(n_coarse)),
        name="base_coarse_constraints",
    )

    for r in range(count):
        divergence = rt_div_vector_list[r]
        model.addConstr(
            gp.quicksum(divergence[i] * w_vec[i] for i in range(coarse_nsqr)) <= v_var,
            name=f"divergence_constraint_{r}",
        )

    if with_bnd:
        b_data, b_indices, b_indptr = b_csr.data, b_csr.indices, b_csr.indptr
        z_bnd = model.addVars(range(b), vtype=GRB.CONTINUOUS, lb=0.0, name="z_bnd")
        for j in range(b):
            start, end = b_indptr[j], b_indptr[j + 1]
            indices = b_indices[start:end]
            coefs = b_data[start:end]
            tv_expr = gp.quicksum(coefs[k] * u[indices[k]] for k in range(len(indices)))
            model.addConstr(z_bnd[j] >= tv_expr, name=f"tv_bnd_pos_{j}")
            model.addConstr(z_bnd[j] >= -tv_expr, name=f"tv_bnd_neg_{j}")

    tv_interior = gp.quicksum(z[j] * hx for j in range(m))
    if with_bnd:
        tv_boundary = gp.quicksum(z_bnd[j] * hx for j in range(b))
        model.addConstr(tv_interior + tv_boundary <= c * v_var, name="tv_bound")
    else:
        model.addConstr(tv_interior <= c * v_var, name="tv_bound")

    model._u = u
    model._V = v_var
    model._w_vec = w_vec
    model._nsqr = coarse_nsqr
    model._h_tv = h_tv
    model._DG0 = dg0_coarse
    model._RT0 = rt0_coarse
    model._M = m_coarse
    model._bnd_dofs = bnd_dofs_coarse
    model._dof_edge_ctrs = dof_edge_ctrs_coarse
    model._nRT0 = nrt0_coarse
    model._base_to_coarse = base_coarse
    model._coarse_nsqr = coarse_nsqr

    model.setParam("OutputFlag", 0)
    model.optimize(lazy_callback)

    if model.status == GRB.OPTIMAL:
        for r in range(count):
            divergence = rt_div_vector_list[r]
            lhs_value = sum(divergence[i] * w_vec[i].X for i in range(coarse_nsqr))
            assert lhs_value - v_var.X <= 1e-6

        u_opt = np.array([u[i].X for i in range(n)], dtype=float)
        tv_interior_value = sum(z[j].X * hx for j in range(m))
        return u_opt, float(v_var.X), float(tv_interior_value)

    return None, None, None


# Backward compatibility for previous API name.
sub_problem_solver_1_cb = solve_mip_subproblem
