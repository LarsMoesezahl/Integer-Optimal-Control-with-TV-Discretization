"""Finite-element PDE model used by the blur benchmark."""

from __future__ import annotations

import copy
import gc
from functools import cached_property

import dolfinx
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import ufl
from dolfinx import fem
from dolfinx.fem import petsc
from matplotlib import cm
from mpi4py import MPI
from petsc4py import PETSc
from scipy.interpolate import griddata

import utility

class BvpBlur:
    def __init__(self, n, nu, f_lbd):
        self.x0 = np.array([ 0.,  0.])
        self.x1 = np.array([ 2.,  2.])
        self.nx = n
        self.ny = n
        self.nu = nu

        self.mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            points=((self.x0, self.x1)),
            n=(self.nx, self.ny),
            cell_type=dolfinx.mesh.CellType.triangle,
            diagonal=dolfinx.mesh.DiagonalType.crossed
        )
        
        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim, self.mesh.topology.dim
        )

        self.x = ufl.SpatialCoordinate(self.mesh)
        self.CG1 = fem.functionspace(self.mesh, ("CG", 1))
        self.V = self.CG1
        self.DG0 = fem.functionspace(self.mesh, ("DG", 0))
        self.CG1_num_dofs = self.CG1.dofmap.index_map.size_global \
            * self.CG1.dofmap.index_map_bs
        self._Pbnd_diag = None
        self._PbndMrhs_csr = None
        self._K_csc = None
        self._Kfull_csc = None
        self._Mrhs_csr = None
        self._K_LUfac_sol = None
        self._full_linpart = None        
        self.w = fem.Function(self.DG0)

        # initialize Dirichlet boundary conditions
        self.setup_bc()
        self.f = fem.Function(self.DG0)
        self.f.interpolate(lambda x: f_lbd(x[0], x[1]))
        self.f_vec = self.f.x.array

        x = ufl.SpatialCoordinate(self.mesh)
        X_ = fem.Function(self.DG0)
        Y_ = fem.Function(self.DG0)
        X_.interpolate(fem.Expression(x[0], self.DG0.element.interpolation_points()))
        Y_.interpolate(fem.Expression(x[1], self.DG0.element.interpolation_points()))
        self.X_gridpts_ctrl_plt = copy.deepcopy(X_.x.array[:])
        self.Y_gridpts_ctrl_plt = copy.deepcopy(Y_.x.array[:])
        self.xgrid_ctrl_plt, self.ygrid_ctrl_plt = np.meshgrid(
            np.linspace(self.x0[0], self.x1[0], self.nx),
            np.linspace(self.x0[1], self.x1[1], self.ny),
        )

        X_ = fem.Function(self.CG1)
        Y_ = fem.Function(self.CG1)
        X_.interpolate(fem.Expression(x[0], self.CG1.element.interpolation_points()))
        Y_.interpolate(fem.Expression(x[1], self.CG1.element.interpolation_points()))
        self.X_gridpts_u_plt = copy.deepcopy(X_.x.array[:])
        self.Y_gridpts_u_plt = copy.deepcopy(Y_.x.array[:])
        self.xgrid_u_plt, self.ygrid_u_plt = np.meshgrid(
            np.linspace(self.x0[0], self.x1[0], self.nx),
            np.linspace(self.x0[1], self.x1[1], self.ny),
        )

    def setup_bc(self):
        facets_ex_bc0 = dolfinx.mesh.locate_entities_boundary(
            self.mesh, dim=(self.mesh.topology.dim - 1),
            marker=lambda x:
                np.logical_or(
                    np.logical_or(
                        np.isclose(x[0], self.x0[0]),
                        np.isclose(x[0], self.x1[0]),
                    ),
                    np.logical_or(
                        np.isclose(x[1], self.x0[1]),
                        np.isclose(x[1], self.x1[1]),
                    )
                )
        )
        self.dofs_ex_bc = fem.locate_dofs_topological(
            V=self.CG1, entity_dim=1, entities=facets_ex_bc0
        )
        self.ex_bc0 = fem.dirichletbc(
            value=PETSc.ScalarType(0.), dofs=self.dofs_ex_bc, V=self.CG1
        )

        self.ex_bc_val = 0.
        self.bc_vec = np.zeros((self.CG1_num_dofs))
        self.bc_vec[self.dofs_ex_bc] = self.ex_bc_val
        g = np.zeros((self.CG1_num_dofs,))
        bc_PKfullg_vec = self.Pbnd_diag @ self.Kfull_csc @ g
        self.bc_vec -= bc_PKfullg_vec

    def solve(self, w):
        # initialize test and trial function
        u = ufl.TrialFunction(self.CG1)
        v = ufl.TestFunction(self.CG1)

        # setup state equation
        lhs = ufl.inner(ufl.grad(v), ufl.grad(u)) * self.nu * ufl.dx \
                + v * u * ufl.dx
        rhs = v * w * ufl.dx + v * self.f * ufl.dx

        # solve state equation
        problem = petsc.LinearProblem(
            lhs, rhs, [self.ex_bc0],
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
        )
        u = problem.solve()
        del problem # need a better solution for this
        gc.collect()
        return u

    @cached_property
    def full_linpart(self):
        if self._full_linpart is None:
            u = ufl.TrialFunction(self.CG1)
            v = ufl.TestFunction(self.CG1)
            self._full_linpart = fem.form(
                ufl.inner(ufl.grad(v), ufl.grad(u)) * self.nu * ufl.dx \
                + v * u * ufl.dx
            )      
        return self._full_linpart

    @cached_property
    def K_csc(self):
        if self._K_csc is None:
            u = ufl.TrialFunction(self.CG1)
            v = ufl.TestFunction(self.CG1)
            self._K_csc = utility.assemble_csc(
                self.full_linpart, [self.ex_bc0]
            )
        return self._K_csc
    
    @cached_property
    def Kfull_csc(self):
        if self._Kfull_csc is None:    
            self._Kfull_csc = utility.assemble_csc(self.full_linpart)
        return self._Kfull_csc

    @cached_property
    def K_LUfac_sol(self):
        if self._K_LUfac_sol is None:
            self._K_LUfac_sol = sp.sparse.linalg.factorized(self.K_csc)
        return self._K_LUfac_sol

    @cached_property
    def Mrhs_csr(self):
        if self._Mrhs_csr is None:
            w = ufl.TrialFunction(self.DG0)
            v = ufl.TestFunction(self.CG1) 
            rhs = fem.form(v * w * ufl.dx)
            self._Mrhs_csr = utility.assemble_csr(rhs)
        return self._Mrhs_csr  

    @cached_property
    def Pbnd_diag(self):             
        if self._Pbnd_diag is None:
            diag = np.ones((self.CG1_num_dofs,))
            diag[self.dofs_ex_bc] = 0.
            self._Pbnd_diag = np.diag(diag)
        return self._Pbnd_diag

    @cached_property
    def PbndMrhs_csr(self):
        if self._PbndMrhs_csr is None:
            self._PbndMrhs_csr = self.Pbnd_diag @ self.Mrhs_csr
        return self._PbndMrhs_csr       

    def assemble_matrices_pde_adj(self, w_vec):
        Mww_b = self.PbndMrhs_csr @ w_vec + self.PbndMrhs_csr @ self.f_vec + self.bc_vec
        u_vec = self.K_LUfac_sol(Mww_b)
        assert np.allclose(u_vec[self.dofs_ex_bc], self.bc_vec[self.dofs_ex_bc])
        return u_vec, self.K_csc, self.PbndMrhs_csr, self.bc_vec

    def plot_w(self, w_vec, title="no title", iter=0, fn_prefix="", fig=None):
        fig = plt.figure(figsize=(5, 5))
        zgrid = griddata(
            (self.X_gridpts_ctrl_plt, self.Y_gridpts_ctrl_plt),
            w_vec,
            (self.xgrid_ctrl_plt, self.ygrid_ctrl_plt),
            method='nearest'
        )
        ax = fig.add_subplot(1, 1, 1)
        surf = ax.contourf(
            self.xgrid_ctrl_plt,
            self.ygrid_ctrl_plt,
            zgrid,
            cmap=cm.binary,
            antialiased=False
        )
        ax.set_title(title)
        fig.tight_layout(pad=5.0)
        plt.savefig(fn_prefix + '_plt_%05u_w.png' % (iter), dpi=600)
        plt.close(fig)        

    def plot_u(self, u_vec, title, iter=0, fn_prefix="", fig=None):
        fig = plt.figure(figsize=(5, 5))
        zgrid = griddata(
            (self.X_gridpts_u_plt, self.Y_gridpts_u_plt),
            u_vec,
            (self.xgrid_u_plt, self.ygrid_u_plt),
            method='cubic'
        )

        ax = fig.add_subplot(1, 1, 1)
        surf = ax.contourf(
            self.xgrid_u_plt,
            self.ygrid_u_plt,
            zgrid,
            cmap=cm.coolwarm
        )
        
        ax.set_title(title)        
        fig.tight_layout(pad=5.0)
        plt.savefig(fn_prefix + '_plt_%05u_u.png' % (iter), dpi=600)
        plt.close(fig)
        return fig

    def plot_u_3d(self, u_vec, title, iter=0, fn_prefix="", fig=None):
        fig = plt.figure(figsize=(5, 5))
        zgrid = griddata(
            (self.X_gridpts_u_plt, self.Y_gridpts_u_plt),
            u_vec,
            (self.xgrid_u_plt, self.ygrid_u_plt),
            method='cubic'
        )

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        kw = {
            'vmin': u_vec.min(),
            'vmax': u_vec.max(),
            'levels': np.linspace(u_vec.min(), u_vec.max(), 100),
}
        surf = ax.contourf(
            self.xgrid_u_plt,
            self.ygrid_u_plt,
            zgrid,
            cmap=cm.coolwarm,
            **kw
        )
        
        ax.set_title(title)        
        fig.tight_layout(pad=5.0)
        plt.savefig(fn_prefix + '_plt_%05u_u.png' % (iter), dpi=600)
        plt.close(fig)
        return fig
