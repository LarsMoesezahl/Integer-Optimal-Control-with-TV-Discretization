"""Tracking-type objective for the blur PDE with TV-regularized controls."""

from __future__ import annotations

import copy

import numpy as np
import scipy as sp

from functools import cached_property

from dolfinx import fem
import ufl

import opt_tv_base
import utility

class OptBlur(opt_tv_base.OptTvBase):
    """Optimization object combining PDE objective and TV regularization."""

    def __init__(self,
                 bvp,
                 alpha,
                 uo_lbd,
                 dir_prefix="",
                 with_bnd=False,
                 do_check_grad=False):
        super().__init__(
            ctrl_domain_xmin=0.,
            ctrl_domain_xmax=2.,
            ctrl_domain_ymin=0.,
            ctrl_domain_ymax=2.,
            with_bnd=with_bnd,
            bvp=bvp,
            fn_prefix = dir_prefix + "outputs/blur"
        )  
        self.alpha = alpha   # scaling of TV term
        self.beta = 0.
        self.bangs = np.array([0., 1.])
        self.do_check_grad = do_check_grad

        # internal variables for pre-assembled PDE matrices
        self._Mw_DG0 = None
        self._Mu4j_V = None
        self._KT_LUfac_sol = None

        # internal variables and parameter values for tracking-type objective
        # functional
        self.uo = fem.Function(self.bvp.V)
        self.uo.interpolate(lambda x: uo_lbd(x[0], x[1]))
        self.uo_vec = self.uo.x.array
   
    @cached_property
    def Mu4j_V(self):
        if self._Mu4j_V is None:
            u = ufl.TrialFunction(self.bvp.V)
            v = ufl.TestFunction(self.bvp.V)
            self._Mu4j_V = utility.assemble_csr(fem.form(u*v*ufl.dx))
        return self._Mu4j_V

    @cached_property
    def KT_LUfac_sol(self):
        if self._KT_LUfac_sol is None:
            self._KT_LUfac_sol = sp.sparse.linalg.factorized(
                self.K_csc.transpose()
            )
        return self._KT_LUfac_sol

    # BELOW: interface for trust-region algorithm

    def eval_j(self, u_vec):
        udiff = u_vec - self.uo_vec
        return .5 * udiff.T @ self.Mu4j_V @ udiff

    def c_from_grad(self, grad):
        return grad

    def f(self, w_vec):
        w_bvp_vec = self.tri_sqr @ w_vec
        usol_vec, _, _, _ = self.bvp.assemble_matrices_pde_adj(w_bvp_vec)
        return self.eval_j(usol_vec)

    def fu(self, w_vec):
        w_bvp_vec = self.tri_sqr @ w_vec
        usol_vec, _, _, _ = self.bvp.assemble_matrices_pde_adj(w_bvp_vec)
        return self.eval_j(usol_vec), usol_vec

    def df(self, w_vec):
        _, g_vec = self.fdf(w_vec, eval_f=False)
        return g_vec

    def fdf(self, w_vec, eval_f=True):
        w_bvp_vec = self.tri_sqr @ w_vec
        usol_vec, self.K_csc, PM_csr, _ = \
            self.bvp.assemble_matrices_pde_adj(w_bvp_vec)

        udiff_vec = copy.deepcopy(usol_vec)
        udiff_vec -= self.uo_vec

        p_vec = self.KT_LUfac_sol(
            self.bvp.Pbnd_diag.T @ (-self.Mu4j_V @ udiff_vec)
        ) 
        assert np.allclose(
            p_vec[self.bvp.dofs_ex_bc],
            np.zeros((self.bvp.dofs_ex_bc.size,)),
        )

        g_vec = -self.bvp.Mrhs_csr.T @ p_vec

        if self.do_check_grad:
            utility.check_grad(self, g_vec, w_bvp_vec)

        g_coarse_vec = self.tri_sqr.T @ g_vec
        if eval_f:
            return self.eval_j(usol_vec), g_coarse_vec
        else:
            return np.nan, g_coarse_vec
