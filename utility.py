"""Shared linear-algebra and FEM helper functions."""

from __future__ import annotations

import copy

import numpy as np
import scipy as sp
import ufl
from dolfinx import fem


def assemble_csr(a, bcs=None):
    """Assemble a bilinear form with boundary conditions into CSR format."""
    if bcs is None:
        bcs = []
    matrix = fem.petsc.create_matrix(a)
    fem.petsc.assemble_matrix(matrix, a, bcs=bcs)
    matrix.assemble()
    return sp.sparse.csr_matrix(matrix.getValuesCSR()[::-1], shape=matrix.size)


def assemble_csc(a, bcs=None):
    """Assemble a bilinear form into CSC format."""
    if bcs is None:
        bcs = []
    return assemble_csr(a, bcs).tocsc()


def assemble_vec(a):
    """Assemble a linear form and return it as a NumPy vector."""
    vector = fem.petsc.create_vector(a)
    fem.petsc.assemble_vector(vector, a)
    vector.assemble()
    return vector.getArray()


def project(uu, v_space):
    """Project a UFL expression `uu` into function space `v_space`."""
    u = ufl.TrialFunction(v_space)
    v = ufl.TestFunction(v_space)
    lhs = u * v * ufl.dx
    rhs = uu * v * ufl.dx
    problem = fem.petsc.LinearProblem(
        lhs,
        rhs,
        bcs=[],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return problem.solve()


def check_grad(opt_obj, grad, w_vec):
    """Finite-difference gradient check for debugging analytical derivatives."""
    loh = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    log = np.zeros((grad.size, len(loh)))
    un_vec, _, _, _ = opt_obj.bvp.assemble_matrices_pde_adj(w_vec)
    jvaln = opt_obj.eval_j(un_vec)
    wh_vec = copy.deepcopy(w_vec)

    for i, h in enumerate(loh):
        print(f"Checking GRAD, FD @ h = {h:.2e}")
        for j in range(wh_vec.size):
            if j % 10000 == 0:
                print(f"{j:8d} of {grad.size:8d}")
            tmp = wh_vec[j]
            wh_vec[j] += h
            uh_vec, _, _, _ = opt_obj.bvp.assemble_matrices_pde_adj(wh_vec)
            jvalh = opt_obj.eval_j(uh_vec)
            log[j, i] = (jvalh - jvaln) / h
            wh_vec[j] = tmp

    for i, _ in enumerate(loh):
        print(
            "%2u   %.4e   %.4e   %.4e"
            % (
                i,
                np.linalg.norm(log[:, i] - grad),
                np.linalg.norm(grad),
                np.linalg.norm(log[:, i]),
            )
        )

        assert np.isclose(
            sp.sparse.linalg.norm(
                sp.sparse.identity(opt_obj.nsqr, format="csr")
                - opt_obj.sqr_tri @ opt_obj.tri_sqr
            ),
            0.0,
        )
        print(f"FVAL = {jvaln:.8e}")
