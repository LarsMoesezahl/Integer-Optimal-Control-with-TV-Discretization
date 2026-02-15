"""Base class providing grid mappings and TV operators for optimization."""

from __future__ import annotations

import copy
import itertools
from functools import cached_property

import dolfinx
import numpy as np
import scipy as sp
from dolfinx import fem
from mpi4py import MPI

class OptTvBase():
    """Base class with TV regularization operators on structured control grids."""

    def __init__(
        self,
        ctrl_domain_xmin,
        ctrl_domain_xmax,
        ctrl_domain_ymin,
        ctrl_domain_ymax,
        with_bnd,   
        bvp,
        fn_prefix,
        merge_factor_x=1,
        merge_factor_y=1
    ):
        self.fn_prefix = fn_prefix

        self.xmin = ctrl_domain_xmin
        self.xmax = ctrl_domain_xmax
        self.ymin = ctrl_domain_ymin
        self.ymax = ctrl_domain_ymax
        self.with_bnd = with_bnd
        self.bvp = bvp
        self.x0 = np.array([ 0.,  0.])
        self.x1 = np.array([ 2.,  2.])
        self.merge_factor_x = merge_factor_x
        self.merge_factor_y = merge_factor_y

        self.pt_bottom_left = np.array(
            [self.xmin, self.ymin]
        )
        self.pt_top_right = np.array(
            [self.xmax, self.ymax]
        )

        self.lenx = self.xmax - self.xmin
        self.leny = self.ymax - self.ymin
        self.nx_max = (self.bvp.nx * self.lenx \
                / (self.bvp.x1[0] - self.bvp.x0[0])).astype(np.int32)
        self.ny_max = (self.bvp.ny * self.leny \
                / (self.bvp.x1[1] - self.bvp.x0[1])).astype(np.int32)

        self.nsqr_max = self.nx_max * self.ny_max        
        self.bvp_cells = dolfinx.mesh.locate_entities(
            self.bvp.mesh, self.bvp.mesh.topology.dim,
            lambda x: np.logical_and(
                np.logical_and(self.xmin <= x[0], x[0] <= self.xmax),
                np.logical_and(self.ymin <= x[1], x[1] <= self.ymax)
            )
        )

        assert self.nx_max % self.merge_factor_x == 0
        assert self.ny_max % self.merge_factor_y == 0

        self.nx = self.nx_max // self.merge_factor_x
        self.ny = self.ny_max // self.merge_factor_y
        self.hx = self.lenx / self.nx
        self.hy = self.leny / self.ny        
        self.nsqr = self.nx * self.ny

        self.bvp_ntriangles = self.bvp_cells.size
        assert self.nsqr_max * 4 == self.bvp_ntriangles
        assert self.nsqr * self.merge_factor_y * self.merge_factor_x == self.nsqr_max

        self.mesh = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            points=((self.pt_bottom_left, self.pt_top_right)),
            n=(self.nx, self.ny),
            cell_type=dolfinx.mesh.CellType.quadrilateral
        )
        
        self.mesh.topology.create_connectivity(
            self.mesh.topology.dim, self.mesh.topology.dim
        )
        
        self.DG0 = fem.functionspace(self.mesh, ("DG", 0))

        # internal variables for relation of control domain to bvp domain
        self._tri_sqr = None
        self._sqr_tri = None
        self._sqr_ctrs_dof = None
        self._sqr_ctrs = None

        # internal variables for tv term computation
        self._c2f = None
        self._f2c = None

        self._B_csr = None
        self._D_csr = None

        # internal variables dof neighbordhood computation for sanity checks
        self._dof_nbh = None

    # functions (and properties) for relation of control domain to bvp domain
    
    @cached_property
    def tri_sqr(self):
        if self._tri_sqr is None:
            self.assemble_tri_sqr_maps()
        return self._tri_sqr

    @cached_property
    def sqr_tri(self):
        if self._sqr_tri is None:
            self.assemble_tri_sqr_maps()
        return self._sqr_tri
    
    @cached_property
    def sqr_ctrs_dof(self):
        if self._sqr_ctrs_dof is None:
            self.assemble_tri_sqr_maps()
        return self._sqr_ctrs_dof    
    
    @cached_property
    def sqr_ctrs(self):
        if self._sqr_ctrs is None:
            self.assemble_tri_sqr_maps()
        return self._sqr_ctrs    
        
    def assemble_tri_sqr_maps(self):
        self._sqr_ctrs = dolfinx.mesh.compute_midpoints(
            self.mesh,
            self.mesh.topology.dim,              
            np.arange(0, self.nsqr, 1).astype(np.int32)
        )
        self._sqr_ctrs_dof = np.zeros((self.nsqr, 2))
        tol = 1e-8
        num_tri = self.bvp.DG0.dofmap.index_map.size_global
        num_tri_per_sqr = 4 * self.merge_factor_y * self.merge_factor_x

        ST_row_idx = np.zeros(4 * self.nsqr_max, dtype=np.int32)
        ST_col_idx = np.zeros(4 * self.nsqr_max, dtype=np.int32)
        ST_vals = np.zeros(4 * self.nsqr_max)
        
        c = 0
        for i_sqr in range(self.nsqr):
            ctr = self._sqr_ctrs[i_sqr]
            tri_cells = dolfinx.mesh.locate_entities(
                self.bvp.mesh,
                self.bvp.mesh.topology.dim,
                lambda x: np.logical_and(
                    np.abs(x[0] - ctr[0]) <= .5 * self.hx + tol,
                    np.abs(x[1] - ctr[1]) <= .5 * self.hy + tol
                )
            )
            sqr_dof = fem.locate_dofs_topological(
                self.DG0,
                self.mesh.topology.dim,
                np.array([i_sqr])
            )[0]

            self._sqr_ctrs_dof[sqr_dof,:] = ctr[:2]
            tri_dofs = fem.locate_dofs_topological(
                self.bvp.DG0,
                self.bvp.mesh.topology.dim,
                tri_cells
            )
            ST_row_idx[c:c + tri_dofs.size] = sqr_dof
            ST_col_idx[c:c + tri_dofs.size] = tri_dofs
            ST_vals[c:c + tri_dofs.size] = 1.
            c += tri_dofs.size
            assert tri_dofs.size == num_tri_per_sqr

        self._sqr_tri = sp.sparse.csr_array(
            (ST_vals, (ST_row_idx, ST_col_idx)),
            shape=(self.nsqr, num_tri)
        )
        self._tri_sqr = copy.deepcopy(self._sqr_tri.T)
        self._sqr_tri /= num_tri_per_sqr

        assert np.sum(
            np.isclose(np.sum(self._tri_sqr, 1), np.ones((num_tri,)))
        ) == self.bvp_ntriangles
        
        assert np.allclose(
            np.sum(self._sqr_tri, 1), np.ones((self.nsqr,))
        )    

    # functions (and properties) for tv term computation

    @cached_property
    def c2f(self):
        if self._c2f is None:
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim,
                self.mesh.topology.dim - 1
            )
            self._c2f = self.mesh.topology.connectivity(
                self.mesh.topology.dim,
                self.mesh.topology.dim - 1
            )
        return self._c2f
    
    @cached_property
    def f2c(self):
        if self._f2c is None:
            self.mesh.topology.create_connectivity(
                self.mesh.topology.dim - 1,
                self.mesh.topology.dim
            )
            self._f2c = self.mesh.topology.connectivity(
                self.mesh.topology.dim - 1,
                self.mesh.topology.dim
            )
        return self._f2c

    @cached_property
    def B_csr(self):
        if self._B_csr is None:
            self.assemble_difference_and_boundary_operators()
        return self._B_csr

    @cached_property
    def D_csr(self):
        if self._D_csr is None:
            self.assemble_difference_and_boundary_operators()
        return self._D_csr
    
    def reduced_operators_from_csr(self, M_csr, dof_subset):
        M_coo = M_csr.tocoo()

        col_idx_subset_mask = np.isin(M_coo.col, dof_subset)

        Mlhs_csr = sp.sparse.csr_array((
            M_coo.data[col_idx_subset_mask],
            (M_coo.row[col_idx_subset_mask], M_coo.col[col_idx_subset_mask])
        ), shape=M_coo.shape)
        Mlhs_csr = Mlhs_csr[:, np.array([vec.nnz for vec in Mlhs_csr.T]) > 0]

        nzrows = np.array([vec.nnz for vec in Mlhs_csr])  > 0
        Mlhs_csr = Mlhs_csr[nzrows]
        Mrhs_csr = M_csr[nzrows]

        return Mlhs_csr, Mrhs_csr

    def assemble_reduced_difference_and_boundary_operators(self, dof_subset):
        Dred_lhs_csr, Dred_rhs_csr = \
            self.reduced_operators_from_csr(self.D_csr, dof_subset)
        Bred_lhs_csr, Bred_rhs_csr = \
            self.reduced_operators_from_csr(self.B_csr, dof_subset)
        return Dred_lhs_csr, Dred_rhs_csr, Bred_lhs_csr, Bred_rhs_csr
   
    def assemble_difference_and_boundary_operators(self):
        facets = dolfinx.mesh.locate_entities(
            self.mesh, 1, lambda x:  np.full((x.shape[1],), True)
        )
        num_facets = facets.size
        bnd_facets = dolfinx.mesh.locate_entities_boundary(self.mesh, 1, 
		        lambda x: np.full((x.shape[1],), True))
        int_facets = np.delete(np.arange(0,num_facets), bnd_facets)
        num_bnd_facets = bnd_facets.size
        num_int_facets = int_facets.size        

        # assemble difference operator
        D_row_idx = np.repeat(np.arange(0, num_int_facets), 2).astype(np.int32)
        D_col_idx = np.zeros((2*num_int_facets,), dtype=np.int32)
        D_val = np.tile(np.array([1.,-1.]), num_int_facets)
        for i, f in enumerate(int_facets):
            sqr_dofs = fem.locate_dofs_topological(
                self.DG0, 2, np.array(self.f2c.links(f).tolist())
            )
            D_col_idx[2*i:2*i+2] = sqr_dofs[:]
            assert sqr_dofs.size == 2
        self._D_csr = sp.sparse.csr_array((D_val, (D_row_idx, D_col_idx)))

        # assemble boundary contribution
        B_row_idx = np.arange(0, num_bnd_facets).astype(np.int32)
        B_col_idx = np.zeros((num_bnd_facets,), dtype=np.int32)
        B_val = np.ones((num_bnd_facets,))
        for i, f in enumerate(bnd_facets):
            sqr_dofs = fem.locate_dofs_topological(
                self.DG0, 2, np.array(self.f2c.links(f).tolist())
            )
            assert sqr_dofs.size == 1
            B_col_idx[i] = sqr_dofs[:]
        self._B_csr = sp.sparse.csr_array((B_val, (B_row_idx, B_col_idx)))

    def eval_tv_interior(self, w_vec):
        return self.D_csr @ w_vec

    def eval_tv_boundary(self, w_vec):
        return self.B_csr @ w_vec
    
    # functions for trust-region algorithm interface

    def save_parameters(self):
        raise NotImplementedError

    def reg(self, w_vec):
        bnd_contribution = np.linalg.norm(self.eval_tv_boundary(w_vec), 1) \
                           if self.with_bnd else 0.
        assert self.hx == self.hy
        return self.hx * (np.linalg.norm(self.eval_tv_interior(w_vec), 1) 
                          + bnd_contribution)

    def get_zero_as_nparray(self):
        return np.zeros((self.nsqr,))

    # functions (and properties) for dof neighbordhood computation for sanity
    # checks

    @cached_property
    def dof_nbh(self):
        if self._dof_nbh is None:            
            self._dof_nbh = []
            for sqr in range(self.nsqr):
                self._dof_nbh += [fem.locate_dofs_topological(
                    self.DG0,
                    self.mesh.topology.dim,
                    np.array(list(itertools.filterfalse(
                        lambda sqr_: sqr_ == sqr,
                        itertools.chain.from_iterable(
                                [self.f2c.links(f).tolist()
                                for f in self.c2f.links(sqr)]
                            )
                        ))
                    )
                )]
        return self._dof_nbh
