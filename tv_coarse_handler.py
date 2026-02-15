"""Coarse-grid TV helper operators for the mixed-integer subproblems."""

from __future__ import annotations

import copy
import itertools
from functools import cached_property

import numpy as np
import scipy as sp

import dolfinx
import ufl
from dolfinx import fem
from mpi4py import MPI

import opt_tv_base as opt_tv_base
import utility


class CoarseTVHandler(opt_tv_base.OptTvBase):
    """Build coarse-grid TV operators and RT0 coupling structures."""

    def __init__(
        self,
        bvp,
        alpha=1e-5,
        dir_prefix=".",
        with_bnd=False,
        do_check_grad=False,
        merge_factor_x=1,
        merge_factor_y=1,
    ):
        super().__init__(
            ctrl_domain_xmin=0.0,
            ctrl_domain_xmax=2.0,
            ctrl_domain_ymin=0.0,
            ctrl_domain_ymax=2.0,
            with_bnd=with_bnd,
            bvp=bvp,
            fn_prefix=dir_prefix + "/ad",
            merge_factor_x=merge_factor_x,
            merge_factor_y=merge_factor_y,
        )

        self.merge_factor_x_coarse = 4
        self.merge_factor_y_coarse = 4

        self.lenx_coarse = self.xmax - self.xmin
        self.leny_coarse = self.ymax - self.ymin
        self.nx_max_coarse = (
            self.nx * self.lenx_coarse / (self.x1[0] - self.x0[0])
        ).astype(np.int32)
        self.ny_max_coarse = (
            self.ny * self.leny_coarse / (self.x1[1] - self.x0[1])
        ).astype(np.int32)

        self.nsqr_max_coarse = (
            self.nx_max_coarse // self.merge_factor_x_coarse
        ) * (self.ny_max_coarse // self.merge_factor_y_coarse)
        self.tv_base_cells = dolfinx.mesh.locate_entities(
            self.mesh,
            self.mesh.topology.dim,
            lambda x: np.logical_and(
                np.logical_and(self.xmin <= x[0], x[0] <= self.xmax),
                np.logical_and(self.ymin <= x[1], x[1] <= self.ymax),
            ),
        )
        # print(f"nx_max_coarse: {self.nx_max_coarse}")
        # print(f"ny_max_coarse: {self.ny_max_coarse}")
        # print(f"merge_factor_x_coarse: {self.merge_factor_x_coarse}")
        # print(f"merge_factor_y_coarse: {self.merge_factor_y_coarse}")
        assert self.nx_max_coarse % self.merge_factor_x_coarse == 0
        assert self.ny_max_coarse % self.merge_factor_y_coarse == 0

        self.nx_coarse = self.nx_max_coarse // self.merge_factor_x_coarse
        self.ny_coarse = self.ny_max_coarse // self.merge_factor_y_coarse
        self.hx_coarse = self.lenx_coarse / self.nx_coarse
        self.hy_coarse = self.leny_coarse / self.ny_coarse
        self.nsqr_coarse = self.nx_coarse * self.ny_coarse

        self.tv_base_nsquares = self.tv_base_cells.size
        assert (
            self.nsqr_max_coarse
            * self.merge_factor_x_coarse
            * self.merge_factor_y_coarse
            == self.tv_base_nsquares
        )

        self.mesh_coarse = dolfinx.mesh.create_rectangle(
            MPI.COMM_WORLD,
            points=(self.pt_bottom_left, self.pt_top_right),
            n=(self.nx_coarse, self.ny_coarse),
            cell_type=dolfinx.mesh.CellType.quadrilateral,
        )
        self.mesh_coarse.topology.create_connectivity(
            self.mesh_coarse.topology.dim, self.mesh_coarse.topology.dim
        )
        self.DG0_coarse = fem.functionspace(self.mesh_coarse, ("DG", 0))
        self.RT0_coarse = fem.functionspace(self.mesh_coarse, ("RT", 1))

        # Internal variables for relation of control domain to bvp domain.
        self._base_coarse = None
        self._coarse_base = None
        self._coarse_ctrs_dof = None
        self._coarse_ctrs = None

        # Internal variables for TV term computation.
        self._c2f_coarse = None
        self._f2c_coarse = None

        self._B_csr_coarse = None
        self._D_csr_coarse = None

        # Internal variables for DOF neighborhood computation for sanity checks.
        self._dof_nbh_coarse = None

        # Functions & internal variables for Raviart–Thomas 0 element
        # optimization for TV computation.
        self.dof_edge_ctrs = []
        w = ufl.TestFunction(self.DG0_coarse)
        phi = ufl.TrialFunction(self.RT0_coarse)
        self.M = utility.assemble_csr(
            fem.form(ufl.inner(w, ufl.div(phi)) * ufl.dx)
        )
        self.nDG0 = self.M.shape[0]
        self.nRT0 = self.M.shape[1]
        self.bnd_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh_coarse, 1, lambda x: np.full((x.shape[1],), True)
        )
        self.bnd_dofs = fem.locate_dofs_topological(
            self.RT0_coarse, 1, self.bnd_facets
        )
        self.dof_edge_ctrs = []
        self.collect_mid_points()

    def get_cell_coordinates(self, cell):
        """Helper function to get the coordinates of a cell in the mesh."""
        connectivity = self.mesh_coarse.topology.connectivity(
            self.mesh_coarse.topology.dim, 0
        )
        assert connectivity is not None, "The connectivity is missing!"

        cell_vertices_indices = connectivity.links(cell)
        coords = self.mesh_coarse.geometry.x[cell_vertices_indices]

        if coords.shape[0] != 4:
            raise ValueError(
                "The cell does not have four vertices. Is the mesh quadrilateral?"
            )

        q1, q2, q3, q4 = coords
        return q1, q2, q3, q4

    def collect_mid_points(self):
        """Helper function to get the midpoints of the edges of the cells in the
        mesh. Ordered along the DOFs of RT0 elements."""
        for cell in range(self.nsqr_coarse):
            q1, q2, q3, q4 = self.get_cell_coordinates(cell)
            left = np.array([0.5 * (q1[0] + q2[0]), 0.5 * (q1[1] + q2[1])])
            bottom = np.array([0.5 * (q1[0] + q3[0]), 0.5 * (q1[1] + q3[1])])
            top = np.array([0.5 * (q2[0] + q4[0]), 0.5 * (q2[1] + q4[1])])
            # Right edge midpoint is the average of the two right-edge vertices.
            right = np.array([0.5 * (q3[0] + q4[0]), 0.5 * (q3[1] + q4[1])])
            self.dof_edge_ctrs.append([left, bottom, top, right])

    # Functions (and properties) for relation of control domain to BVP domain.

    @cached_property
    def coarse_base(self):
        if self._base_coarse is None:
            self.assemble_base_coarse_maps()
        return self._base_coarse

    @cached_property
    def base_coarse(self):
        if self._coarse_base is None:
            self.assemble_base_coarse_maps()
        return self._coarse_base

    @cached_property
    def coarse_ctrs_dof(self):
        if self._coarse_ctrs_dof is None:
            self.assemble_base_coarse_maps()
        return self._coarse_ctrs_dof

    @cached_property
    def coarse_ctrs(self):
        if self._coarse_ctrs is None:
            self.assemble_base_coarse_maps()
        return self._coarse_ctrs

    def assemble_base_coarse_maps(self):
        self._coarse_ctrs = dolfinx.mesh.compute_midpoints(
            self.mesh_coarse,
            self.mesh_coarse.topology.dim,
            np.arange(0, self.nsqr_coarse, 1).astype(np.int32),
        )
        self._coarse_ctrs_dof = np.zeros((self.nsqr_coarse, 2))
        tol = 1e-8
        num_base_sqr = self.DG0.dofmap.index_map.size_global
        num_base_sqr_per_csqr = (
            self.merge_factor_y_coarse * self.merge_factor_x_coarse
        )

        total_size = (
            self.merge_factor_x_coarse
            * self.merge_factor_y_coarse
            * self.nsqr_max_coarse
        )
        ST_row_idx = np.zeros(total_size, dtype=np.int32)
        ST_col_idx = np.zeros(total_size, dtype=np.int32)
        ST_vals = np.zeros(total_size)

        c = 0
        for i_sqr in range(self.nsqr_coarse):
            ctr = self._coarse_ctrs[i_sqr]
            base_cells = dolfinx.mesh.locate_entities(
                self.mesh,
                self.mesh.topology.dim,
                lambda x: np.logical_and(
                    np.abs(x[0] - ctr[0]) <= 0.5 * self.hx_coarse + tol,
                    np.abs(x[1] - ctr[1]) <= 0.5 * self.hy_coarse + tol,
                ),
            )
            coarse_dof = fem.locate_dofs_topological(
                self.DG0_coarse,
                self.mesh_coarse.topology.dim,
                np.array([i_sqr]),
            )[0]
            self._coarse_ctrs_dof[coarse_dof, :] = ctr[:2]
            base_dofs = fem.locate_dofs_topological(
                self.DG0, self.mesh.topology.dim, base_cells
            )
            ST_row_idx[c : c + base_dofs.size] = coarse_dof
            ST_col_idx[c : c + base_dofs.size] = base_dofs
            ST_vals[c : c + base_dofs.size] = 1.0
            c += base_dofs.size
            assert base_dofs.size == num_base_sqr_per_csqr

        self._coarse_base = sp.sparse.csr_array(
            (ST_vals, (ST_row_idx, ST_col_idx)),
            shape=(self.nsqr_coarse, num_base_sqr),
        )
        self._base_coarse = copy.deepcopy(self._coarse_base.T)
        self._coarse_base /= num_base_sqr_per_csqr

        assert np.sum(
            np.isclose(np.sum(self._base_coarse, 1), np.ones((num_base_sqr,)))
        ) == self.tv_base_nsquares
        assert np.allclose(
            np.sum(self._coarse_base, 1), np.ones((self.nsqr_coarse,))
        )

    # Functions (and properties) for TV term computation.

    @cached_property
    def c2f_coarse(self):
        if self._c2f_coarse is None:
            self.mesh_coarse.topology.create_connectivity(
                self.mesh_coarse.topology.dim,
                self.mesh_coarse.topology.dim - 1,
            )
            self._c2f_coarse = self.mesh_coarse.topology.connectivity(
                self.mesh_coarse.topology.dim,
                self.mesh_coarse.topology.dim - 1,
            )
        return self._c2f_coarse

    @cached_property
    def f2c_coarse(self):
        if self._f2c_coarse is None:
            self.mesh_coarse.topology.create_connectivity(
                self.mesh_coarse.topology.dim - 1,
                self.mesh_coarse.topology.dim,
            )
            self._f2c_coarse = self.mesh_coarse.topology.connectivity(
                self.mesh_coarse.topology.dim - 1,
                self.mesh_coarse.topology.dim,
            )
        return self._f2c_coarse

    @cached_property
    def B_csr_coarse(self):
        if self._B_csr_coarse is None:
            self.assemble_difference_and_boundary_operators_coarse()
        return self._B_csr_coarse

    @cached_property
    def D_csr_coarse(self):
        if self._D_csr_coarse is None:
            self.assemble_difference_and_boundary_operators_coarse()
        return self._D_csr_coarse

    def reduced_operators_from_csr(self, M_csr, dof_subset):
        M_coo = M_csr.tocoo()
        col_idx_subset_mask = np.isin(M_coo.col, dof_subset)
        Mlhs_csr = sp.sparse.csr_array(
            (
                M_coo.data[col_idx_subset_mask],
                (M_coo.row[col_idx_subset_mask], M_coo.col[col_idx_subset_mask]),
            ),
            shape=M_coo.shape,
        )
        Mlhs_csr = Mlhs_csr[:, Mlhs_csr.getnnz(0) > 0]
        nzrows = Mlhs_csr.getnnz(1) > 0
        Mlhs_csr = Mlhs_csr[nzrows]
        Mrhs_csr = M_csr[nzrows]
        return Mlhs_csr, Mrhs_csr

    def assemble_reduced_difference_and_boundary_operators_coarse(self, dof_subset):
        Dred_lhs_csr, Dred_rhs_csr = self.reduced_operators_from_csr(
            self.D_csr_coarse, dof_subset
        )
        Bred_lhs_csr, Bred_rhs_csr = self.reduced_operators_from_csr(
            self.B_csr_coarse, dof_subset
        )
        return Dred_lhs_csr, Dred_rhs_csr, Bred_lhs_csr, Bred_rhs_csr

    def assemble_difference_and_boundary_operators_coarse(self):
        facets = dolfinx.mesh.locate_entities(
            self.mesh_coarse, 1, lambda x: np.full((x.shape[1],), True)
        )
        num_facets = facets.size
        bnd_facets = dolfinx.mesh.locate_entities_boundary(
            self.mesh_coarse, 1, lambda x: np.full((x.shape[1],), True)
        )
        int_facets = np.delete(np.arange(0, num_facets), bnd_facets)
        num_bnd_facets = bnd_facets.size
        num_int_facets = int_facets.size

        # Assemble difference operator.
        D_row_idx = np.repeat(np.arange(0, num_int_facets), 2).astype(np.int32)
        D_col_idx = np.zeros(2 * num_int_facets, dtype=np.int32)
        D_val = np.tile(np.array([1.0, -1.0]), num_int_facets)
        for i, f in enumerate(int_facets):
            sqr_dofs = fem.locate_dofs_topological(
                self.DG0_coarse,
                2,
                np.array(self.f2c_coarse.links(f).tolist()),
            )
            D_col_idx[2 * i : 2 * i + 2] = sqr_dofs[:]
            assert sqr_dofs.size == 2
        self._D_csr_coarse = sp.sparse.csr_array((D_val, (D_row_idx, D_col_idx)))

        # Assemble boundary contribution.
        B_row_idx = np.arange(0, num_bnd_facets).astype(np.int32)
        B_col_idx = np.zeros(num_bnd_facets, dtype=np.int32)
        B_val = np.ones(num_bnd_facets)
        for i, f in enumerate(bnd_facets):
            sqr_dofs = fem.locate_dofs_topological(
                self.DG0_coarse,
                2,
                np.array(self.f2c_coarse.links(f).tolist()),
            )
            assert sqr_dofs.size == 1
            B_col_idx[i] = sqr_dofs[0]
        self._B_csr_coarse = sp.sparse.csr_array((B_val, (B_row_idx, B_col_idx)))

    def eval_tv_interior_coarse(self, w_vec):
        return self.D_csr_coarse @ (self.base_coarse.T @ w_vec)

    def eval_tv_boundary_coarse(self, w_vec):
        return self.B_csr_coarse @ (self.base_coarse @ w_vec)

    # Functions for trust-region algorithm interface.

    def save_parameters(self):
        raise NotImplementedError

    def reg_coarse(self, w_vec):
        bnd_contribution = (
            np.linalg.norm(self.eval_tv_boundary_coarse(w_vec), 1)
            if self.with_bnd
            else 0.0
        )
        assert self.hx_coarse == self.hy_coarse
        return self.hx_coarse * (
            np.linalg.norm(self.eval_tv_interior_coarse(w_vec), 1) + bnd_contribution
        )

    def get_zero_as_nparray(self):
        return np.zeros((self.nsqr_coarse,))


    @cached_property
    def dof_nbh_coarse(self):
        if self._dof_nbh_coarse is None:
            self._dof_nbh_coarse = []
            for sqr in range(self.nsqr):
                nbhs = fem.locate_dofs_topological(
                    self.DG0_coarse,
                    self.mesh_coarse.topology.dim,
                    np.array(
                        list(
                            itertools.filterfalse(
                                lambda sqr_: sqr_ == sqr,
                                itertools.chain.from_iterable(
                                    [
                                        self.f2c_coarse.links(f).tolist()
                                        for f in self.c2f_coarse.links(sqr)
                                    ]
                                ),
                            )
                        )
                    ),
                )
                self._dof_nbh_coarse.append(nbhs)
        return self._dof_nbh_coarse


# Backward compatibility for previous API name.
coarse_tv_handler = CoarseTVHandler
