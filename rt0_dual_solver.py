"""Dual RT0 subproblem used for TV-related cuts and diagnostics."""

from __future__ import annotations

from typing import Sequence

import gurobipy as grb
import numpy as np


def get_cell_coordinates(xgrid, cell: int):
    """Return the four vertex coordinates of a quadrilateral cell."""
    connectivity = xgrid.topology.connectivity(xgrid.topology.dim, 0)
    if connectivity is None:
        raise ValueError("Mesh connectivity (cell-to-vertex) is missing.")

    cell_vertices_indices = connectivity.links(cell)
    coords = xgrid.geometry.x[cell_vertices_indices]
    if coords.shape[0] != 4:
        raise ValueError("Expected quadrilateral cells with four vertices.")

    q1, q2, q3, q4 = coords
    return q1, q2, q3, q4


def mid_points(nsqr: int, xgrid) -> list[list[np.ndarray]]:
    """Return edge midpoint coordinates for all cells.

    Midpoints are ordered as `[left, bottom, top, right]` to match RT0 dof
    conventions used throughout the project.
    """
    edge_midpoints: list[list[np.ndarray]] = []
    for cell in range(nsqr):
        q1, q2, q3, q4 = get_cell_coordinates(xgrid, cell)
        left = np.array([0.5 * (q1[0] + q2[0]), 0.5 * (q1[1] + q2[1])])
        bottom = np.array([0.5 * (q1[0] + q3[0]), 0.5 * (q1[1] + q3[1])])
        top = np.array([0.5 * (q2[0] + q4[0]), 0.5 * (q2[1] + q4[1])])
        right = np.array([0.5 * (q3[0] + q4[0]), 0.5 * (q3[1] + q4[1])])
        edge_midpoints.append([left, bottom, top, right])
    return edge_midpoints


def solve_rt0_divergence_subproblem(
    nsqr: int,
    h: float,
    w_vec: np.ndarray,
    dg0,
    rt0,
    m_matrix,
    bnd_dofs: Sequence[int],
    dof_edge_ctrs: Sequence[Sequence[np.ndarray]],
    nrt0: int,
    base_to_coarse,
):
    """Solve the RT0 dual problem and return divergence + TV value."""
    del dg0  # Kept in signature for API compatibility.

    w_vec = base_to_coarse @ w_vec
    c_vec = np.transpose(m_matrix.T @ w_vec)

    opt_model = grb.Model("DRT0")
    opt_phi = opt_model.addMVar(
        nrt0,
        lb=np.full((nrt0,), -1000.0).tolist(),
        ub=np.full((nrt0,), 1000.0).tolist(),
        obj=c_vec.tolist(),
    )

    for dof in bnd_dofs:
        opt_model.addConstr(opt_phi[dof] == 0.0)

    for i in range(nsqr):
        dofs = rt0.dofmap.cell_dofs(i)
        assert dofs.ndim == 1 and dofs.shape[0] == 4

        d30x = dof_edge_ctrs[i][3][0] - dof_edge_ctrs[i][0][0]
        d21y = dof_edge_ctrs[i][2][1] - dof_edge_ctrs[i][1][1]
        assert np.isclose(d30x, h) and np.isclose(d21y, h)

        opt_model.addConstr(
            h**-2 * opt_phi[dofs[0]] ** 2 + h**-2 * opt_phi[dofs[2]] ** 2 <= 1.0
        )
        opt_model.addConstr(
            h**-2 * opt_phi[dofs[3]] ** 2 + h**-2 * opt_phi[dofs[2]] ** 2 <= 1.0
        )
        opt_model.addConstr(
            h**-2 * opt_phi[dofs[0]] ** 2 + h**-2 * opt_phi[dofs[1]] ** 2 <= 1.0
        )
        opt_model.addConstr(
            h**-2 * opt_phi[dofs[3]] ** 2 + h**-2 * opt_phi[dofs[1]] ** 2 <= 1.0
        )

    opt_model.setParam("OutputFlag", 0)
    opt_model.optimize()

    div_phi = m_matrix @ opt_phi.x[:]
    tv_from_opt = -np.dot(div_phi, w_vec)
    return div_phi, tv_from_opt


# Backward compatibility for previous API name.
div_tr0_solver = solve_rt0_divergence_subproblem
