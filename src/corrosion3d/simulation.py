"""Three-dimensional field container and weak-form mechanical operators."""

import numpy as np
import pyamg
from scipy import ndimage
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .core import generate_microstructure_3d


class CorrosionCracking3D:
    def __init__(
        self,
        seed=88,
        grid_spacing=None,
        domain_thickness=None,
        domain_width=None,
        domain_height=None,
        itz_width=0.003,
        aggregate_records=None,
        surface_gauge_length=0.02,
        steel_interface_width=None,
        surface_observation_depth=None,
        rebar_radius=0.01,
        cover_depth=0.05,
        axial_symmetry_ends=True,
    ):
        self.Dc_mat_base = 0.0001
        self.K_b_binding = 1.0
        self.C_max_surf = 4.0
        self.cover_depth = float(cover_depth)
        self.C_critical = 1.2
        self.k_rust_rate = 0.00015
        self.delta_0_gap = 8e-06
        self.K_langmuir = 0.35
        self.i_corr_ref = 0.01
        self.i_corr_passive = 0.0002
        self.i_corr_max = 0.2
        self.rust_expansion_ratio = 2.5
        self.corrosion_damage_feedback = 2.0
        self.seconds_per_year = 365.25 * 24 * 3600
        self.faraday = 96485.33212
        self.M_fe = 0.055845
        self.rho_fe = 7850.0
        self.n_fe = 2.0
        self.Gc_mat_base = 80.0
        self.E_mat_base = 25000000000.0
        self.random_seed = int(seed)
        self.itz_width = float(itz_width)
        self.aggregate_records = aggregate_records
        self.surface_gauge_length = float(surface_gauge_length)
        self.steel_interface_width = (
            None if steel_interface_width is None else float(steel_interface_width)
        )
        self.surface_observation_depth = (
            None
            if surface_observation_depth is None
            else float(surface_observation_depth)
        )
        self.axial_symmetry_ends = bool(axial_symmetry_ends)
        self.W = 0.12 if domain_width is None else float(domain_width)
        self.H = 0.2 if domain_height is None else float(domain_height)
        self.D = 0.04 if domain_thickness is None else float(domain_thickness)
        if min(self.W, self.H, self.D) <= 0.0:
            raise ValueError("domain dimensions must be positive")
        if grid_spacing is None:
            self.Nx, self.Ny, self.Nz = (49, 81, 17)
        else:
            h = float(grid_spacing)
            self.Nx = int(round(self.W / h)) + 1
            self.Ny = int(round(self.H / h)) + 1
            self.Nz = int(round(self.D / h)) + 1
        self.dx, self.dy, self.dz = (
            self.W / (self.Nx - 1),
            self.H / (self.Ny - 1),
            self.D / (self.Nz - 1),
        )
        self.N = self.Nx * self.Ny * self.Nz
        self.shape_3d = (self.Nx, self.Ny, self.Nz)
        self.X, self.Y, self.Z = np.meshgrid(
            np.linspace(0, self.W, self.Nx),
            np.linspace(0, self.H, self.Ny),
            np.linspace(0, self.D, self.Nz),
            indexing="ij",
        )
        self.R_bar = float(rebar_radius)
        self.cx = self.W / 2.0
        self.cy = self.H - self.cover_depth - self.R_bar
        self.Dw_mat, self.Dw_agg, self.Dw_itz = (0.002, 1e-12, 0.008)
        self.kappa, self.V_scale, self.beta = (20000.0, 1.5, 80.0)
        self.agg_vol_frac = 0.3
        self.nu, self.l0, self.k_res = (0.2, 0.005, 1e-06)
        self.S_init = 0.1
        self.S = np.full(self.shape_3d, self.S_init)
        self.C, self.d, self.H_hist = (
            np.zeros(self.shape_3d),
            np.zeros(self.shape_3d),
            np.zeros(self.shape_3d),
        )
        self.u, self.v, self.w = (
            np.zeros(self.shape_3d),
            np.zeros(self.shape_3d),
            np.zeros(self.shape_3d),
        )
        self.x_corr = np.zeros(self.shape_3d)
        self.i_corr = np.zeros(self.shape_3d)
        self.u_r = np.zeros(self.shape_3d)
        self.Na = np.full(self.shape_3d, 60.0)
        self.OH = np.full(self.shape_3d, 80.0)
        self.electric_grad = np.zeros(self.shape_3d)
        self.grad_phi_x = np.zeros(self.shape_3d)
        self.grad_phi_y = np.zeros(self.shape_3d)
        self.grad_phi_z = np.zeros(self.shape_3d)
        self.rust_generated = np.zeros(self.shape_3d)
        self.rust_volume = np.zeros(self.shape_3d)
        self.rust_mobile = np.zeros(self.shape_3d)
        self.rust_compacted = np.zeros(self.shape_3d)
        self.rust_filled = np.zeros(self.shape_3d)
        self.rust_lost = np.zeros(self.shape_3d)
        self.residual_wedge = np.zeros(self.shape_3d)
        self.rust_pressure = np.zeros(self.shape_3d)
        self.rust_bulk_modulus = 2000000000.0
        self.rust_transport_loss_coeff = 0.15
        self.crack_storage_coeff = 2.5e-05
        self.t_ini_map = np.full(self.shape_3d, np.inf)
        self.t_record, self.max_ur_record, self.d_max_record = ([], [], [])
        self.video_frames = []
        self.elastic_energy_j = 0.0
        self.fracture_energy_j = 0.0
        self.pressure_work_j = 0.0
        self.surface_crack_width_mm = 0.0
        self.surface_crack_width_displacement_mm = 0.0
        self.surface_crack_width_envelope_mm = 0.0
        self.surface_crack_width_scale = 1.0
        self.surface_gauge_locked = False
        self.surface_gauge_center_x = np.nan
        self.surface_gauge_center_z = np.nan
        self.surface_gauge_normal_x = np.nan
        self.surface_gauge_normal_z = np.nan
        self.mechanics_rtol = 1e-06
        self.mechanics_max_iterations = 1000
        self.last_mechanics_residual = np.inf
        self.last_mechanics_iterations = 0
        self.last_mechanics_converged = False
        self.last_contact_min_separation_m = 0.0
        self.last_contact_complementarity_residual = 0.0
        self.last_contact_linear_info = 0
        self.last_contact_linear_iterations = 0
        self.last_contact_active_iterations = 0
        self.last_contact_active_cells = 0
        self.last_contact_pdas_fallback_used = False
        self._mechanics_preconditioners = None
        self._mechanics_preconditioner_damage = None
        self._mechanics_direct_preconditioner = None
        self._mechanics_direct_factor = None
        self._mechanics_direct_preconditioner_damage = None
        self._mechanics_direct_preconditioner_material_signature = None
        self._mechanics_stiffness = None
        self._mechanics_stiffness_damage = None
        self._mechanics_stiffness_material_signature = None
        self._mechanics_preconditioner_stiffness = None
        self._mechanics_rigid_modes = None
        self._mechanics_rigid_penalty = None
        self.mechanics_anchor_mask = np.zeros(self.shape_3d, dtype=bool)
        self._generate_meso_structure_3d()
        self.K_bulk = self.E_map / (3 * (1 - 2 * self.nu))
        self.mu_map = self.E_map / (2 * (1 + self.nu))
        self.lam_map = self.K_bulk - 2.0 / 3.0 * self.mu_map

    def _generate_meso_structure_3d(self):
        dist_xy = np.sqrt((self.X - self.cx) ** 2 + (self.Y - self.cy) ** 2)
        self.is_rebar = dist_xy <= self.R_bar
        micro = generate_microstructure_3d(
            shape=self.shape_3d,
            size=(self.W, self.H, self.D),
            rebar_radius=self.R_bar,
            cover_depth=self.cover_depth,
            target_aggregate_fraction=self.agg_vol_frac,
            itz_width=self.itz_width,
            seed=self.random_seed,
            max_attempts=50000,
            records=self.aggregate_records,
        )
        self.is_agg = micro.aggregate.copy()
        self.is_itz = micro.itz.copy()
        self.itz_fraction = micro.itz_fraction.copy()
        self.is_rebar = micro.rebar.copy()
        self.is_agg[self.is_rebar] = False
        self.is_itz[self.is_agg | self.is_rebar] = False
        self.itz_fraction[self.is_agg | self.is_rebar] = 0.0
        if self.steel_interface_width is None:
            interface_structure = ndimage.generate_binary_structure(3, 2)
            self.rebar_bnd = (
                ndimage.binary_dilation(
                    self.is_rebar, structure=interface_structure, iterations=1
                )
                & ~self.is_rebar
                & ~self.is_agg
            )
            self.rebar_interface_fraction = self.rebar_bnd.astype(float)
        else:
            # Cell-overlap fractions represent one physical interface annulus.
            radial_offset = dist_xy - self.R_bar
            radial_cell_size = max(self.dx, self.dy)
            overlap_lower = np.maximum(
                radial_offset - 0.5 * radial_cell_size, 0.0
            )
            overlap_upper = np.minimum(
                radial_offset + 0.5 * radial_cell_size,
                self.steel_interface_width,
            )
            self.rebar_interface_fraction = np.clip(
                (overlap_upper - overlap_lower) / radial_cell_size, 0.0, 1.0
            )
            self.rebar_bnd = (
                (radial_offset > 0.0)
                & (self.rebar_interface_fraction > 0.0)
                & ~self.is_agg
            )
            self.rebar_interface_fraction[~self.rebar_bnd] = 0.0
        if not np.any(self.rebar_bnd):
            raise ValueError(
                "the prescribed steel-interface width contains no grid nodes"
            )
        self.nx, self.ny = (np.zeros_like(self.X), np.zeros_like(self.Y))
        sd = np.maximum(dist_xy, 1e-12)
        self.nx[self.rebar_bnd] = (self.X - self.cx)[self.rebar_bnd] / sd[
            self.rebar_bnd
        ]
        self.ny[self.rebar_bnd] = (self.Y - self.cy)[self.rebar_bnd] / sd[
            self.rebar_bnd
        ]
        self._build_rebar_interface_quadrature()
        self.agg_list = micro.records
        self.slice_polygons = []
        self.Dc_map = np.where(
            self.is_agg, 1e-12, self.Dc_mat_base * (1.0 + 4.0 * self.itz_fraction)
        )
        self.Dw_map = np.where(
            self.is_agg,
            1e-12,
            self.Dw_mat + (self.Dw_itz - self.Dw_mat) * self.itz_fraction,
        )
        self.Gc_map = np.where(
            self.is_agg,
            self.Gc_mat_base * 10.0,
            self.Gc_mat_base * (1.0 - 0.6 * self.itz_fraction),
        )
        self.E_map = np.where(
            self.is_agg,
            60000000000.0,
            self.E_mat_base * (1.0 - 0.4 * self.itz_fraction),
        )
        self.actual_aggregate_fraction = micro.actual_aggregate_fraction
        self.actual_itz_fraction = micro.actual_itz_fraction
        return

    def _build_rebar_interface_quadrature(self):
        self.rebar_area_weight = np.zeros(self.shape_3d)
        self.rebar_vector_area_x = np.zeros(self.shape_3d)
        self.rebar_vector_area_y = np.zeros(self.shape_3d)
        xy_nodes = np.argwhere(np.any(self.rebar_bnd, axis=2))
        if xy_nodes.size == 0:
            return
        angles = np.mod(
            np.arctan2(
                xy_nodes[:, 1] * self.dy - self.cy, xy_nodes[:, 0] * self.dx - self.cx
            ),
            2.0 * np.pi,
        )
        order = np.argsort(angles)
        xy_nodes = xy_nodes[order]
        angles = angles[order]
        z_weights = np.full(self.Nz, self.dz)
        if self.Nz > 1:
            z_weights[[0, -1]] *= 0.5
        else:
            z_weights[:] = self.D
        for index, ((ix, iy), angle) in enumerate(zip(xy_nodes, angles)):
            previous_angle = angles[index - 1]
            next_angle = angles[(index + 1) % len(angles)]
            if index == 0:
                previous_angle -= 2.0 * np.pi
            if index == len(angles) - 1:
                next_angle += 2.0 * np.pi
            lower = 0.5 * (previous_angle + angle)
            upper = 0.5 * (angle + next_angle)
            active_z = np.flatnonzero(self.rebar_bnd[ix, iy, :])
            if active_z.size == 0:
                continue
            local_z_weights = z_weights[active_z]
            local_z_weights *= self.D / max(float(np.sum(local_z_weights)), 1e-30)
            scalar_per_length = self.R_bar * (upper - lower)
            vector_x_per_length = self.R_bar * (np.sin(upper) - np.sin(lower))
            vector_y_per_length = self.R_bar * (-np.cos(upper) + np.cos(lower))
            self.rebar_area_weight[ix, iy, active_z] = (
                scalar_per_length * local_z_weights
            )
            self.rebar_vector_area_x[ix, iy, active_z] = (
                vector_x_per_length * local_z_weights
            )
            self.rebar_vector_area_y[ix, iy, active_z] = (
                vector_y_per_length * local_z_weights
            )
        return

    def build_harmonic_laplacian_3d(self, K, axial_transport=True):
        eps = 1e-20
        KXp, KXm, KYp, KYm, KZp, KZm = [np.zeros_like(K) for _ in range(6)]
        KXp[:-1, :, :] = (
            2 * K[:-1, :, :] * K[1:, :, :] / (K[:-1, :, :] + K[1:, :, :] + eps)
        )
        KXm[1:, :, :] = (
            2 * K[:-1, :, :] * K[1:, :, :] / (K[:-1, :, :] + K[1:, :, :] + eps)
        )
        KYp[:, :-1, :] = (
            2 * K[:, :-1, :] * K[:, 1:, :] / (K[:, :-1, :] + K[:, 1:, :] + eps)
        )
        KYm[:, 1:, :] = (
            2 * K[:, :-1, :] * K[:, 1:, :] / (K[:, :-1, :] + K[:, 1:, :] + eps)
        )
        KZp[:, :, :-1] = (
            2 * K[:, :, :-1] * K[:, :, 1:] / (K[:, :, :-1] + K[:, :, 1:] + eps)
        )
        KZm[:, :, 1:] = (
            2 * K[:, :, :-1] * K[:, :, 1:] / (K[:, :, :-1] + K[:, :, 1:] + eps)
        )
        KXp[-1, :, :] = 0
        KXm[0, :, :] = 0
        KYp[:, -1, :] = 0
        KYm[:, 0, :] = 0
        if axial_transport:
            KZp[:, :, -1] = 0
            KZm[:, :, 0] = 0
        else:
            KZp[:] = 0.0
            KZm[:] = 0.0
        Axp, Axm = ((KXp / self.dx**2).flatten(), (KXm / self.dx**2).flatten())
        Ayp, Aym = ((KYp / self.dy**2).flatten(), (KYm / self.dy**2).flatten())
        Azp, Azm = ((KZp / self.dz**2).flatten(), (KZm / self.dz**2).flatten())
        Ac = -(Axp + Axm + Ayp + Aym + Azp + Azm)
        off_x, off_y, off_z = (self.Ny * self.Nz, self.Nz, 1)
        return sp.diags(
            [
                Axm[off_x:],
                Aym[off_y:],
                Azm[off_z:],
                Ac,
                Azp[:-off_z],
                Ayp[:-off_y],
                Axp[:-off_x],
            ],
            [-off_x, -off_y, -off_z, 0, off_z, off_y, off_x],
            shape=(self.N, self.N),
            format="csr",
        )

    def grad_3d(self, arr, step, axis):
        g = np.zeros_like(arr)
        if axis == 0:
            g[1:-1, :, :] = (arr[2:, :, :] - arr[:-2, :, :]) / (2 * step)
            g[0, :, :] = (arr[1, :, :] - arr[0, :, :]) / step
            g[-1, :, :] = (arr[-1, :, :] - arr[-2, :, :]) / step
        elif axis == 1:
            g[:, 1:-1, :] = (arr[:, 2:, :] - arr[:, :-2, :]) / (2 * step)
            g[:, 0, :] = (arr[:, 1, :] - arr[:, 0, :]) / step
            g[:, -1, :] = (arr[:, -1, :] - arr[:, -2, :]) / step
        else:
            g[:, :, 1:-1] = (arr[:, :, 2:] - arr[:, :, :-2]) / (2 * step)
            g[:, :, 0] = (arr[:, :, 1] - arr[:, :, 0]) / step
            g[:, :, -1] = (arr[:, :, -1] - arr[:, :, -2]) / step
        return g

    def weak_form_strain_components(self):
        solid = ~self.is_rebar

        def forward_gradient(field, step, axis):
            gradient = np.zeros_like(field)
            valid = np.zeros(self.shape_3d, dtype=bool)
            if axis == 0:
                valid[:-1, :, :] = solid[:-1, :, :] & solid[1:, :, :]
                gradient[:-1, :, :] = (field[1:, :, :] - field[:-1, :, :]) / step
            elif axis == 1:
                valid[:, :-1, :] = solid[:, :-1, :] & solid[:, 1:, :]
                gradient[:, :-1, :] = (field[:, 1:, :] - field[:, :-1, :]) / step
            else:
                valid[:, :, :-1] = solid[:, :, :-1] & solid[:, :, 1:]
                gradient[:, :, :-1] = (field[:, :, 1:] - field[:, :, :-1]) / step
            gradient[~valid] = 0.0
            return gradient

        ux = forward_gradient(self.u, self.dx, 0)
        uy = forward_gradient(self.u, self.dy, 1)
        uz = forward_gradient(self.u, self.dz, 2)
        vx = forward_gradient(self.v, self.dx, 0)
        vy = forward_gradient(self.v, self.dy, 1)
        vz = forward_gradient(self.v, self.dz, 2)
        wx = forward_gradient(self.w, self.dx, 0)
        wy = forward_gradient(self.w, self.dy, 1)
        wz = forward_gradient(self.w, self.dz, 2)
        return (ux, vy, wz, 0.5 * (uy + vx), 0.5 * (uz + wx), 0.5 * (vz + wy))

    def solve_mechanics_pseudo_transient(
        self,
        ur_map=None,
        pressure_map=None,
        update_history=True,
        unilateral_contact_active=None,
        unilateral_contact_wedge=None,
        unilateral_contact_stiffness=None,
    ):
        if ur_map is None:
            ur_map = self.u_r
        if pressure_map is None:
            pressure_map = self.rust_pressure
        contact_active = None
        if unilateral_contact_active is not None:
            contact_active = (
                np.asarray(unilateral_contact_active, dtype=bool)
                & self.rebar_bnd
            )
            if np.any(contact_active):
                contact_pressure = np.zeros_like(self.rust_pressure)
                contact_pressure[contact_active] = (
                    max(float(unilateral_contact_stiffness), 0.0)
                    * np.asarray(unilateral_contact_wedge, dtype=float)[contact_active]
                )
                pressure_map = contact_pressure
            else:
                contact_active = None
        deg = (1.0 - self.d) ** 2 + self.k_res
        lam_d, mu_d = (self.lam_map * deg, self.mu_map * deg)
        pressure_force_x = np.zeros_like(self.u)
        pressure_force_y = np.zeros_like(self.v)
        # Signed pressure is needed for matrix-free contact-compliance actions;
        # physical contact calls still provide non-negative pressure.
        active_pressure = self.rebar_bnd & (np.abs(pressure_map) > 0.0)
        if not np.any(active_pressure):
            self.u[:] = 0.0
            self.v[:] = 0.0
            self.w[:] = 0.0
            self.last_mechanics_residual = 0.0
            self.last_mechanics_iterations = 0
            self.last_mechanics_converged = True
            return
        cell_volume = self.dx * self.dy * self.dz
        pressure_force_x[active_pressure] = (
            pressure_map[active_pressure]
            * self.rebar_vector_area_x[active_pressure]
            / cell_volume
        )
        pressure_force_y[active_pressure] = (
            pressure_map[active_pressure]
            * self.rebar_vector_area_y[active_pressure]
            / cell_volume
        )
        fixed_u = self.is_rebar.copy()
        fixed_v = self.is_rebar.copy()
        fixed_w = self.is_rebar.copy()
        if self.axial_symmetry_ends:
            fixed_w[:, :, 0] = True
            fixed_w[:, :, -1] = True
        self.mechanics_anchor_mask[:] = False
        fixed = np.concatenate((fixed_u.ravel(), fixed_v.ravel(), fixed_w.ravel()))
        size = 3 * self.N
        matvec_calls = [0]
        if self._mechanics_rigid_modes is None:
            solid_flat = (~self.is_rebar).ravel().astype(float)
            x = self.X.ravel() - self.cx
            y = self.Y.ravel() - self.cy
            rigid_candidates = np.zeros((size, 3), dtype=float)
            rigid_candidates[: self.N, 0] = solid_flat
            rigid_candidates[self.N : 2 * self.N, 1] = solid_flat
            rigid_candidates[: self.N, 2] = -y * solid_flat
            rigid_candidates[self.N : 2 * self.N, 2] = x * solid_flat
            rigid_candidates[fixed, :] = 0.0
            rigid_modes, rigid_scale = np.linalg.qr(rigid_candidates, mode="reduced")
            independent = np.abs(np.diag(rigid_scale)) > 1e-12
            self._mechanics_rigid_modes = rigid_modes[:, independent]
        rigid_modes = self._mechanics_rigid_modes
        preconditioner_fixed = fixed.copy()
        anchor_a = np.ravel_multi_index((0, 0, 0), self.shape_3d)
        anchor_b = np.ravel_multi_index((self.Nx - 1, 0, 0), self.shape_3d)
        preconditioner_fixed[anchor_a] = True
        preconditioner_fixed[self.N + anchor_a] = True
        preconditioner_fixed[self.N + anchor_b] = True
        solid = ~self.is_rebar
        valid_x = np.zeros(self.shape_3d, dtype=bool)
        valid_y = np.zeros(self.shape_3d, dtype=bool)
        valid_z = np.zeros(self.shape_3d, dtype=bool)
        valid_x[:-1, :, :] = solid[:-1, :, :] & solid[1:, :, :]
        valid_y[:, :-1, :] = solid[:, :-1, :] & solid[:, 1:, :]
        valid_z[:, :, :-1] = solid[:, :, :-1] & solid[:, :, 1:]
        valid_edges = (valid_x, valid_y, valid_z)

        def forward_gradient(field, step, axis):
            gradient = np.zeros_like(field)
            if axis == 0:
                gradient[:-1, :, :] = (field[1:, :, :] - field[:-1, :, :]) / step
            elif axis == 1:
                gradient[:, :-1, :] = (field[:, 1:, :] - field[:, :-1, :]) / step
            else:
                gradient[:, :, :-1] = (field[:, :, 1:] - field[:, :, :-1]) / step
            gradient[~valid_edges[axis]] = 0.0
            return gradient

        def gradient_transpose(field, step, axis):
            field = np.where(valid_edges[axis], field, 0.0)
            result = np.zeros_like(field)
            if axis == 0:
                result[:-1, :, :] -= field[:-1, :, :] / step
                result[1:, :, :] += field[:-1, :, :] / step
            elif axis == 1:
                result[:, :-1, :] -= field[:, :-1, :] / step
                result[:, 1:, :] += field[:, :-1, :] / step
            else:
                result[:, :, :-1] -= field[:, :, :-1] / step
                result[:, :, 1:] += field[:, :, :-1] / step
            return result

        def internal_operator(vector):
            matvec_calls[0] += 1
            u = vector[: self.N].reshape(self.shape_3d).copy()
            v = vector[self.N : 2 * self.N].reshape(self.shape_3d).copy()
            w = vector[2 * self.N :].reshape(self.shape_3d).copy()
            u[fixed_u] = 0.0
            v[fixed_v] = 0.0
            w[fixed_w] = 0.0
            ux = forward_gradient(u, self.dx, 0)
            uy = forward_gradient(u, self.dy, 1)
            uz = forward_gradient(u, self.dz, 2)
            vx = forward_gradient(v, self.dx, 0)
            vy = forward_gradient(v, self.dy, 1)
            vz = forward_gradient(v, self.dz, 2)
            wx = forward_gradient(w, self.dx, 0)
            wy = forward_gradient(w, self.dy, 1)
            wz = forward_gradient(w, self.dz, 2)
            trace = ux + vy + wz
            sxx = lam_d * trace + 2.0 * mu_d * ux
            syy = lam_d * trace + 2.0 * mu_d * vy
            szz = lam_d * trace + 2.0 * mu_d * wz
            txy = mu_d * (uy + vx)
            txz = mu_d * (uz + wx)
            tyz = mu_d * (vz + wy)
            out_u = (
                gradient_transpose(sxx, self.dx, 0)
                + gradient_transpose(txy, self.dy, 1)
                + gradient_transpose(txz, self.dz, 2)
            )
            out_v = (
                gradient_transpose(txy, self.dx, 0)
                + gradient_transpose(syy, self.dy, 1)
                + gradient_transpose(tyz, self.dz, 2)
            )
            out_w = (
                gradient_transpose(txz, self.dx, 0)
                + gradient_transpose(tyz, self.dy, 1)
                + gradient_transpose(szz, self.dz, 2)
            )
            result = np.concatenate((out_u.ravel(), out_v.ravel(), out_w.ravel()))
            result[fixed] = vector[fixed]
            return result

        rhs = np.concatenate(
            (pressure_force_x.ravel(), pressure_force_y.ravel(), np.zeros(self.N))
        )
        rhs[fixed] = 0.0
        if not hasattr(self, "_mechanics_gradient_matrices"):

            def gradient_matrix(valid, offset, step):
                rows = np.flatnonzero(valid.ravel())
                data = np.concatenate(
                    (-np.full(rows.size, 1.0 / step), np.full(rows.size, 1.0 / step))
                )
                return sp.coo_matrix(
                    (
                        data,
                        (
                            np.concatenate((rows, rows)),
                            np.concatenate((rows, rows + offset)),
                        ),
                    ),
                    shape=(self.N, self.N),
                ).tocsr()

            self._mechanics_gradient_matrices = (
                gradient_matrix(valid_x, self.Ny * self.Nz, self.dx),
                gradient_matrix(valid_y, self.Nz, self.dy),
                gradient_matrix(valid_z, 1, self.dz),
            )
        gx, gy, gz = self._mechanics_gradient_matrices

        def weighted_product(left, coefficient, right):
            return left.T @ right.multiply(coefficient.reshape((-1, 1)))

        material_signature = (id(self.lam_map), id(self.mu_map))
        rebuild_stiffness = (
            self._mechanics_stiffness is None
            or self._mechanics_stiffness_damage is None
            or self._mechanics_stiffness_material_signature != material_signature
            or (not np.array_equal(self.d, self._mechanics_stiffness_damage))
        )
        if rebuild_stiffness:
            lam_flat = lam_d.ravel()
            mu_flat = mu_d.ravel()
            normal_flat = lam_flat + 2.0 * mu_flat
            auu = (
                weighted_product(gx, normal_flat, gx)
                + weighted_product(gy, mu_flat, gy)
                + weighted_product(gz, mu_flat, gz)
            )
            avv = (
                weighted_product(gx, mu_flat, gx)
                + weighted_product(gy, normal_flat, gy)
                + weighted_product(gz, mu_flat, gz)
            )
            aww = (
                weighted_product(gx, mu_flat, gx)
                + weighted_product(gy, mu_flat, gy)
                + weighted_product(gz, normal_flat, gz)
            )
            auv = weighted_product(gx, lam_flat, gy) + weighted_product(gy, mu_flat, gx)
            auw = weighted_product(gx, lam_flat, gz) + weighted_product(gz, mu_flat, gx)
            avw = weighted_product(gy, lam_flat, gz) + weighted_product(gz, mu_flat, gy)
            stiffness = sp.bmat(
                [[auu, auv, auw], [auv.T, avv, avw], [auw.T, avw.T, aww]], format="csr"
            )
            free_diagonal = sp.diags((~fixed).astype(float), format="csr")
            reference_diagonal = max(float(np.max(np.abs(stiffness.diagonal()))), 1.0)
            stiffness = free_diagonal @ stiffness @ free_diagonal + sp.diags(
                fixed.astype(float) * reference_diagonal, format="csr"
            )
            preconditioner_free = sp.diags(
                (~preconditioner_fixed).astype(float), format="csr"
            )
            preconditioner_stiffness = (
                preconditioner_free @ stiffness @ preconditioner_free
                + sp.diags(
                    preconditioner_fixed.astype(float) * reference_diagonal,
                    format="csr",
                )
            )
            self._mechanics_stiffness = stiffness
            self._mechanics_preconditioner_stiffness = preconditioner_stiffness
            self._mechanics_rigid_penalty = reference_diagonal
            self._mechanics_stiffness_damage = self.d.copy()
            self._mechanics_stiffness_material_signature = material_signature
        else:
            stiffness = self._mechanics_stiffness
            preconditioner_stiffness = self._mechanics_preconditioner_stiffness
        solve_stiffness = stiffness
        if contact_active is not None:
            spring = np.zeros(self.N, dtype=float)
            spring[contact_active.ravel()] = (
                max(float(unilateral_contact_stiffness), 0.0)
                * self.rebar_area_weight[contact_active]
                / cell_volume
            )
            nx_flat = self.nx.ravel()
            ny_flat = self.ny.ravel()
            kxx = sp.diags(spring * nx_flat**2, format="csr")
            kxy = sp.diags(spring * nx_flat * ny_flat, format="csr")
            kyy = sp.diags(spring * ny_flat**2, format="csr")
            zero_block = sp.csr_matrix((self.N, self.N))
            contact_matrix = sp.bmat(
                [
                    [kxx, kxy, zero_block],
                    [kxy, kyy, zero_block],
                    [zero_block, zero_block, zero_block],
                ],
                format="csr",
            )
            free_diagonal = sp.diags((~fixed).astype(float), format="csr")
            solve_stiffness = (
                stiffness + free_diagonal @ contact_matrix @ free_diagonal
            )
        rigid_penalty = self._mechanics_rigid_penalty

        def counted_matvec(vector):
            matvec_calls[0] += 1
            return solve_stiffness @ vector + rigid_penalty * rigid_modes @ (
                rigid_modes.T @ vector
            )

        operator = spla.LinearOperator((size, size), matvec=counted_matvec, dtype=float)
        use_direct_preconditioner = size <= 40000
        if use_direct_preconditioner:
            rebuild_direct_preconditioner = (
                self._mechanics_direct_preconditioner is None
                or self._mechanics_direct_preconditioner_damage is None
                or self._mechanics_direct_preconditioner_material_signature
                != material_signature
                or (
                    float(
                        np.max(
                            np.abs(
                                self.d - self._mechanics_direct_preconditioner_damage
                            )
                        )
                    )
                    > 0.08
                )
            )
            if rebuild_direct_preconditioner:
                direct_factor = spla.splu(preconditioner_stiffness.tocsc())
                self._mechanics_direct_factor = direct_factor
                anchor_only = preconditioner_fixed & ~fixed

                def projected_direct_solve(vector):
                    rigid_component = rigid_modes @ (rigid_modes.T @ vector)
                    balanced = vector - rigid_component
                    balanced[anchor_only] = 0.0
                    constrained_solution = direct_factor.solve(balanced)
                    constrained_solution -= rigid_modes @ (
                        rigid_modes.T @ constrained_solution
                    )
                    return constrained_solution + rigid_component / rigid_penalty

                self._mechanics_direct_preconditioner = spla.LinearOperator(
                    (size, size), matvec=projected_direct_solve, dtype=float
                )
                self._mechanics_direct_preconditioner_damage = self.d.copy()
                self._mechanics_direct_preconditioner_material_signature = (
                    material_signature
                )
            preconditioner = self._mechanics_direct_preconditioner
        else:
            rebuild_preconditioner = (
                self._mechanics_preconditioners is None
                or self._mechanics_preconditioner_damage is None
                or float(np.max(np.abs(self.d - self._mechanics_preconditioner_damage)))
                > 0.02
            )
            if rebuild_preconditioner:
                x = self.X.ravel() - self.cx
                y = self.Y.ravel() - self.cy
                z = self.Z.ravel() - 0.5 * self.D
                near_null = np.zeros((size, 6))
                near_null[: self.N, 0] = 1.0
                near_null[self.N : 2 * self.N, 1] = 1.0
                near_null[2 * self.N :, 2] = 1.0
                near_null[self.N : 2 * self.N, 3] = -z
                near_null[2 * self.N :, 3] = y
                near_null[: self.N, 4] = z
                near_null[2 * self.N :, 4] = -x
                near_null[: self.N, 5] = -y
                near_null[self.N : 2 * self.N, 5] = x
                near_null[preconditioner_fixed, :] = 0.0
                hierarchy = pyamg.smoothed_aggregation_solver(
                    preconditioner_stiffness,
                    B=near_null,
                    symmetry="symmetric",
                    max_coarse=100,
                )
                self._mechanics_preconditioners = hierarchy.aspreconditioner(cycle="V")
                self._mechanics_preconditioner_damage = self.d.copy()
            preconditioner = self._mechanics_preconditioners
        initial = np.concatenate((self.u.ravel(), self.v.ravel(), self.w.ravel()))
        initial[fixed] = 0.0
        rhs -= rigid_modes @ (rigid_modes.T @ rhs)
        initial -= rigid_modes @ (rigid_modes.T @ initial)
        direct_factor_is_exact = bool(
            contact_active is None
            and use_direct_preconditioner
            and self._mechanics_direct_factor is not None
            and (self._mechanics_direct_preconditioner_damage is not None)
            and (
                self._mechanics_direct_preconditioner_material_signature
                == material_signature
            )
            and np.array_equal(self.d, self._mechanics_direct_preconditioner_damage)
        )
        if direct_factor_is_exact:
            solution = preconditioner @ rhs
            info = 0
        else:
            solution, info = spla.cg(
                operator,
                rhs,
                x0=initial,
                rtol=self.mechanics_rtol,
                atol=0.0,
                maxiter=self.mechanics_max_iterations,
                M=preconditioner,
            )
        residual = operator @ solution - rhs
        free = ~fixed
        residual_norm = float(
            np.linalg.norm(residual[free]) / max(np.linalg.norm(rhs[free]), 1.0)
        )
        if info != 0 or residual_norm > 5.0 * self.mechanics_rtol:
            solution, info = spla.lgmres(
                operator,
                rhs,
                x0=solution,
                rtol=self.mechanics_rtol,
                atol=0.0,
                maxiter=min(self.mechanics_max_iterations, 100),
                M=preconditioner,
                inner_m=50,
                outer_k=5,
            )
            residual = operator @ solution - rhs
            residual_norm = float(
                np.linalg.norm(residual[free]) / max(np.linalg.norm(rhs[free]), 1.0)
            )
        self.last_mechanics_residual = residual_norm
        self.last_mechanics_iterations = matvec_calls[0]
        self.last_mechanics_converged = bool(
            info == 0 and residual_norm <= 5.0 * self.mechanics_rtol
        )
        if not self.last_mechanics_converged:
            raise RuntimeError(
                f"mechanical equilibrium failed: info={info}, relative residual={residual_norm:.3e}, operator calls={matvec_calls[0]}"
            )
        self.u[:] = solution[: self.N].reshape(self.shape_3d)
        self.v[:] = solution[self.N : 2 * self.N].reshape(self.shape_3d)
        self.w[:] = solution[2 * self.N :].reshape(self.shape_3d)
        self.u[fixed_u] = 0.0
        self.v[fixed_v] = 0.0
        self.w[fixed_w] = 0.0
        if not update_history:
            return
        exx, eyy, ezz, exy, exz, eyz = self.weak_form_strain_components()
        eps = np.empty(self.shape_3d + (3, 3), dtype=float)
        eps[..., 0, 0] = exx
        eps[..., 1, 1] = eyy
        eps[..., 2, 2] = ezz
        eps[..., 0, 1] = eps[..., 1, 0] = exy
        eps[..., 0, 2] = eps[..., 2, 0] = exz
        eps[..., 1, 2] = eps[..., 2, 1] = eyz
        principal = np.linalg.eigvalsh(eps)
        pos = np.maximum(principal, 0.0)
        tr_pos = np.maximum(np.sum(principal, axis=-1), 0.0)
        psi_plus = 0.5 * self.lam_map * tr_pos**2 + self.mu_map * np.sum(
            pos**2, axis=-1
        )
        self.H_hist = np.maximum(self.H_hist, psi_plus)
        phase_locked = self.is_rebar | self.mechanics_anchor_mask
        if self.axial_symmetry_ends:
            phase_locked[:, :, 0] = True
            phase_locked[:, :, -1] = True
        self.H_hist[phase_locked] = 0.0

    def update_electric_field(self):
        cl_mol = 1000.0 * np.maximum(self.C, 0.0) / 35.45
        na_mol = np.maximum(self.Na, 1e-12)
        oh_mol = np.maximum(self.OH, 1e-12)
        d_cl_x, d_cl_y, d_cl_z = (
            self.grad_3d(cl_mol, self.dx, 0),
            self.grad_3d(cl_mol, self.dy, 1),
            self.grad_3d(cl_mol, self.dz, 2),
        )
        d_na_x, d_na_y, d_na_z = (
            self.grad_3d(na_mol, self.dx, 0),
            self.grad_3d(na_mol, self.dy, 1),
            self.grad_3d(na_mol, self.dz, 2),
        )
        d_oh_x, d_oh_y, d_oh_z = (
            self.grad_3d(oh_mol, self.dx, 0),
            self.grad_3d(oh_mol, self.dy, 1),
            self.grad_3d(oh_mol, self.dz, 2),
        )
        d_cl = self.Dc_map
        d_na = 0.65 * self.Dc_map
        d_oh = 2.6 * self.Dc_map
        denominator = d_cl * cl_mol + d_na * na_mol + d_oh * oh_mol + 1e-30
        rt_over_f = 8.314462618 * 293.15 / 96485.33212
        gx = -rt_over_f * (-d_cl * d_cl_x + d_na * d_na_x - d_oh * d_oh_x) / denominator
        gy = -rt_over_f * (-d_cl * d_cl_y + d_na * d_na_y - d_oh * d_oh_y) / denominator
        gz = -rt_over_f * (-d_cl * d_cl_z + d_na * d_na_z - d_oh * d_oh_z) / denominator
        self.grad_phi_x = gx
        self.grad_phi_y = gy
        self.grad_phi_z = gz
        self.electric_grad = np.sqrt(gx**2 + gy**2 + gz**2)
        self.electric_grad[self.is_rebar | self.is_agg] = 0.0
        self.grad_phi_x[self.is_rebar | self.is_agg] = 0.0
        self.grad_phi_y[self.is_rebar | self.is_agg] = 0.0
        self.grad_phi_z[self.is_rebar | self.is_agg] = 0.0
        return self.electric_grad

    def update_energy_diagnostics(self):
        exx, eyy, ezz, exy, exz, eyz = self.weak_form_strain_components()
        trace = exx + eyy + ezz
        deg = (1.0 - self.d) ** 2 + self.k_res
        elastic_density = deg * (
            0.5 * self.lam_map * trace**2
            + self.mu_map
            * (exx**2 + eyy**2 + ezz**2 + 2.0 * (exy**2 + exz**2 + eyz**2))
        )
        elastic_density[self.is_rebar] = 0.0
        volume = self.dx * self.dy * self.dz
        self.elastic_energy_j = float(np.sum(np.maximum(elastic_density, 0.0)) * volume)
        solid = ~self.is_rebar
        local_fracture = np.where(
            solid, self.Gc_map * self.d**2 / (2.0 * max(self.l0, 1e-12)), 0.0
        )
        fracture_energy = float(np.sum(local_fracture) * volume)
        phase_coefficient = self.Gc_map * self.l0
        eps = 1e-30
        for axis, step in enumerate((self.dx, self.dy, self.dz)):
            lower = [slice(None), slice(None), slice(None)]
            upper = [slice(None), slice(None), slice(None)]
            lower[axis] = slice(None, -1)
            upper[axis] = slice(1, None)
            lower = tuple(lower)
            upper = tuple(upper)
            valid = solid[lower] & solid[upper]
            coefficient = (
                2.0
                * phase_coefficient[lower]
                * phase_coefficient[upper]
                / (phase_coefficient[lower] + phase_coefficient[upper] + eps)
            )
            jump = (self.d[upper] - self.d[lower]) / step
            fracture_energy += float(
                0.5 * volume * np.sum(coefficient[valid] * jump[valid] ** 2)
            )
        self.fracture_energy_j = fracture_energy
        normal_displacement = self.u * self.nx + self.v * self.ny
        self.pressure_work_j = float(
            np.sum(
                self.rust_pressure[self.rebar_bnd]
                * np.maximum(normal_displacement[self.rebar_bnd], 0.0)
                * self.rebar_area_weight[self.rebar_bnd]
            )
        )
        reference_energy = max(
            abs(self.pressure_work_j), 2.0 * abs(self.elastic_energy_j), 1.0e-30
        )
        self.mechanical_energy_relative_error = float(
            abs(self.pressure_work_j - 2.0 * self.elastic_energy_j)
            / reference_energy
        )
        current_width = self.estimate_surface_crack_width_mm()
        self.surface_crack_width_mm = float(current_width)
        self.surface_crack_width_envelope_mm = max(
            float(self.surface_crack_width_envelope_mm), float(current_width)
        )
        return (self.elastic_energy_j, self.fracture_energy_j, self.pressure_work_j)

    def diagnostic_tensile_strain_and_driving_energy(self):
        """Return the resolved maximum tensile strain and phase-field drive."""
        exx, eyy, ezz, exy, exz, eyz = self.weak_form_strain_components()
        eps = np.empty(self.shape_3d + (3, 3), dtype=float)
        eps[..., 0, 0] = exx
        eps[..., 1, 1] = eyy
        eps[..., 2, 2] = ezz
        eps[..., 0, 1] = eps[..., 1, 0] = exy
        eps[..., 0, 2] = eps[..., 2, 0] = exz
        eps[..., 1, 2] = eps[..., 2, 1] = eyz
        principal = np.linalg.eigvalsh(eps)
        max_tensile_strain = float(np.max(np.maximum(principal[..., -1], 0.0)))
        max_driving_energy = float(np.max(self.H_hist))
        return max_tensile_strain, max_driving_energy

    def diagnostic_damage_component(self, threshold=0.2):
        """Return largest damage component size and whether it reaches the exposed surface."""
        mask = (self.d >= float(threshold)) & ~self.is_rebar
        connectivity = ndimage.generate_binary_structure(3, 1)
        labels, count = ndimage.label(mask, structure=connectivity)
        if count == 0:
            return 0, 0.0, False
        sizes = np.bincount(labels.ravel())[1:]
        largest = int(np.max(sizes))
        largest_label = int(np.argmax(sizes) + 1)
        component = labels == largest_label
        touches_surface = bool(np.any(component[:, -1, :]))
        return largest, largest * self.dx * self.dy * self.dz, touches_surface

    def diagnostic_spanning_damage_component(self, threshold=0.2):
        """Return the largest component touching both rebar interface and surface."""
        mask = (self.d >= float(threshold)) & ~self.is_rebar
        connectivity = ndimage.generate_binary_structure(3, 1)
        labels, count = ndimage.label(mask, structure=connectivity)
        if count == 0:
            return 0, 0.0, False
        largest = 0
        for label in range(1, count + 1):
            component = labels == label
            touches_interface = bool(np.any(component & self.rebar_bnd))
            touches_surface = bool(np.any(component[:, -1, :]))
            if touches_interface and touches_surface:
                largest = max(largest, int(np.count_nonzero(component)))
        return largest, largest * self.dx * self.dy * self.dz, largest > 0

    def _surface_gauge_opening(self, center_x, center_z, normal_x, normal_z):
        half_gauge = 0.5 * self.surface_gauge_length
        x_minus = center_x - half_gauge * normal_x
        x_plus = center_x + half_gauge * normal_x
        z_minus = center_z - half_gauge * normal_z
        z_plus = center_z + half_gauge * normal_z
        if min(x_minus, x_plus, z_minus, z_plus) < 0.0:
            return None
        if max(x_minus, x_plus) > self.W or max(z_minus, z_plus) > self.D:
            return None
        coordinates = np.array(
            [
                [x_minus / self.dx, x_plus / self.dx],
                [self.Ny - 1.0, self.Ny - 1.0],
                [z_minus / self.dz, z_plus / self.dz],
            ]
        )
        u_pair = ndimage.map_coordinates(self.u, coordinates, order=1, mode="nearest")
        w_pair = ndimage.map_coordinates(self.w, coordinates, order=1, mode="nearest")
        return max(
            float(
                (u_pair[1] - u_pair[0]) * normal_x + (w_pair[1] - w_pair[0]) * normal_z
            ),
            0.0,
        )

    def _scan_surface_gauges(self, damage_cutoff=0.0):
        """Scan displacement gauges independently of the localization threshold."""
        damage_cutoff = max(float(damage_cutoff), 0.0)
        if self.surface_observation_depth is None:
            surface_rows = np.arange(max(self.Ny - 2, 0), self.Ny)
        else:
            y0 = self.H - self.surface_observation_depth
            surface_rows = np.flatnonzero(self.Y[0, :, 0] >= y0 - 1.0e-12)
            if surface_rows.size == 0:
                surface_rows = np.array([self.Ny - 1])
        surface_damage = np.max(self.d[:, surface_rows, :], axis=1)
        candidates = np.argwhere(surface_damage >= damage_cutoff)
        if candidates.size == 0:
            return (0.0, None)
        damage_values = surface_damage[candidates[:, 0], candidates[:, 1]]
        candidates = candidates[np.argsort(damage_values)[-min(64, len(candidates)) :]]
        angles = np.linspace(0.0, np.pi, 12, endpoint=False)
        max_opening = 0.0
        best_gauge = None
        for ix, iz in candidates:
            center_x = ix * self.dx
            center_z = iz * self.dz
            for angle in angles:
                normal_x = float(np.cos(angle))
                normal_z = float(np.sin(angle))
                opening = self._surface_gauge_opening(
                    center_x, center_z, normal_x, normal_z
                )
                if opening is None:
                    continue
                if opening > max_opening:
                    max_opening = opening
                    best_gauge = (center_x, center_z, normal_x, normal_z)
        return (max_opening, best_gauge)

    def estimate_surface_crack_width_mm(self):
        if self.surface_gauge_locked:
            opening = self._surface_gauge_opening(
                self.surface_gauge_center_x,
                self.surface_gauge_center_z,
                self.surface_gauge_normal_x,
                self.surface_gauge_normal_z,
            )
            max_opening = 0.0 if opening is None else opening
        else:
            max_opening, best_gauge = self._scan_surface_gauges()
            if 1000.0 * max_opening >= 0.05 and best_gauge is not None:
                (
                    self.surface_gauge_center_x,
                    self.surface_gauge_center_z,
                    self.surface_gauge_normal_x,
                    self.surface_gauge_normal_z,
                ) = best_gauge
                self.surface_gauge_locked = True
        self.surface_crack_width_displacement_mm = 1000.0 * max_opening
        return float(
            self.surface_crack_width_scale * self.surface_crack_width_displacement_mm
        )
