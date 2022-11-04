from dolfinx import mesh, fem, jit, io
from mpi4py import MPI
from utils import reorder_mesh, norm_L2, domain_average
from dolfinx.cpp.mesh import cell_num_entities
import numpy as np
from ufl import inner, grad, dot, div
import ufl
from dolfinx.fem import IntegralType
import cffi
import numba
from petsc4py import PETSc
from dolfinx.cpp.fem import Form_float64
from dolfinx_hdg.assemble import assemble_matrix_block as assemble_matrix_block_hdg
from dolfinx_hdg.assemble import assemble_vector_block as assemble_vector_block_hdg
from dolfinx_hdg.assemble import pack_coefficients
import sys


def boundary(x):
    lr = np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
    tb = np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
    lrtb = lr | tb
    if tdim == 2:
        return lrtb
    else:
        assert tdim == 3
        fb = np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)
        return lrtb | fb


comm = MPI.COMM_WORLD
rank = comm.rank


def par_print(string):
    if rank == 0:
        print(string)
        sys.stdout.flush()


par_print("Create mesh")
n = 8
msh = mesh.create_unit_square(
    comm, n, n, ghost_mode=mesh.GhostMode.none)
# msh = mesh.create_unit_cube(
#     comm, n, n, n, ghost_mode=mesh.GhostMode.none)

par_print("Reorder mesh")
# Currently, permutations are not working in parallel, so reorder the
# mesh
reorder_mesh(msh)

par_print("Create facet mesh")
tdim = msh.topology.dim
fdim = tdim - 1

num_cell_facets = cell_num_entities(msh.topology.cell_type, fdim)
msh.topology.create_entities(fdim)
facet_imap = msh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
facets = np.arange(num_facets, dtype=np.int32)

# NOTE Despite all facets being present in the submesh, the entity map isn't
# necessarily the identity in parallel
facet_mesh, entity_map = mesh.create_submesh(msh, fdim, facets)[0:2]

par_print("Create function spaces")
k = 2
V = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k))
Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k - 1))
Vbar = fem.VectorFunctionSpace(
    facet_mesh, ("Discontinuous Lagrange", k))
Qbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

par_print("Define problem")
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)
ubar = ufl.TrialFunction(Vbar)
vbar = ufl.TestFunction(Vbar)
pbar = ufl.TrialFunction(Qbar)
qbar = ufl.TestFunction(Qbar)

V_ele_space_dim = V.element.space_dimension
Vbar_ele_space_dim = Vbar.element.space_dimension
Q_ele_space_dim = Q.element.space_dimension
Qbar_ele_space_dim = Qbar.element.space_dimension

num_cell_facets = msh.ufl_cell().num_facets()
num_dofs_g = len(msh.geometry.dofmap.links(0))

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)

gamma = k**2 / h
if tdim == 2:
    gamma *= 6.0
else:
    assert tdim == 3
    gamma *= 10.0


def u_e(x, module=np):
    if tdim == 2:
        u_x = module.sin(module.pi * x[0]) * module.cos(module.pi * x[1])
        u_y = - module.sin(module.pi * x[1]) * module.cos(module.pi * x[0])
        if module == np:
            return np.stack((u_x, u_y))
        else:
            return ufl.as_vector((u_x, u_y))
    else:
        assert tdim == 3
        u_x = module.sin(module.pi * x[0]) * module.cos(module.pi * x[1]) - \
            module.sin(module.pi * x[0]) * module.cos(module.pi * x[2])
        u_y = module.sin(module.pi * x[1]) * module.cos(module.pi * x[2]) - \
            module.sin(module.pi * x[1]) * module.cos(module.pi * x[0])
        u_z = module.sin(module.pi * x[2]) * module.cos(module.pi * x[0]) - \
            module.sin(module.pi * x[2]) * module.cos(module.pi * x[1])
        if module == np:
            return np.stack((u_x, u_y, u_z))
        else:
            return ufl.as_vector((u_x, u_y, u_z))


def p_e(x, module=np):
    if tdim == 2:
        return module.sin(module.pi * x[0]) * module.sin(module.pi * x[1])
    else:
        assert tdim == 3
        return module.sin(module.pi * x[0]) * module.sin(module.pi * x[1]) * module.sin(module.pi * x[2])


dx_c = ufl.Measure("dx", domain=msh)
ds_c = ufl.Measure("ds", domain=msh)

x = ufl.SpatialCoordinate(msh)
f = - div(grad(u_e(x, ufl))) + grad(p_e(x, ufl))

a_00 = inner(grad(u), grad(v)) * dx_c + gamma * inner(u, v) * ds_c \
    - (inner(u, dot(grad(v), n))
       + inner(v, dot(grad(u), n))) * ds_c
a_10 = - inner(q, div(u)) * dx_c
a_20 = inner(vbar, dot(grad(u), n)) * ds_c - gamma * inner(vbar, u) * ds_c
a_30 = inner(dot(u, n), qbar) * ds_c
a_22 = gamma * inner(ubar, vbar) * ds_c

p_11 = h * inner(pbar, qbar) * ds_c

L_0 = inner(f, v) * dx_c

par_print("Create inverse entity map")
inv_entity_map = np.full_like(entity_map, -1)
for i, f in enumerate(entity_map):
    inv_entity_map[f] = i

par_print("JIT kernels")
nptype = "float64"
ffcxtype = "double"

# LHS forms
ufcx_form_00, _, _ = jit.ffcx_jit(
    msh.comm, a_00, form_compiler_options={"scalar_type": ffcxtype})
kernel_00_cell = getattr(ufcx_form_00.integrals(IntegralType.cell)[0],
                         f"tabulate_tensor_{nptype}")
kernel_00_facet = getattr(ufcx_form_00.integrals(IntegralType.exterior_facet)[0],
                          f"tabulate_tensor_{nptype}")
ufcx_form_10, _, _ = jit.ffcx_jit(
    msh.comm, a_10, form_compiler_options={"scalar_type": ffcxtype})
kernel_10 = getattr(ufcx_form_10.integrals(IntegralType.cell)[0],
                    f"tabulate_tensor_{nptype}")
ufcx_form_20, _, _ = jit.ffcx_jit(
    msh.comm, a_20, form_compiler_options={"scalar_type": ffcxtype})
kernel_20 = getattr(ufcx_form_20.integrals(IntegralType.exterior_facet)[0],
                    f"tabulate_tensor_{nptype}")
ufcx_form_30, _, _ = jit.ffcx_jit(
    msh.comm, a_30, form_compiler_options={"scalar_type": ffcxtype})
kernel_30 = getattr(ufcx_form_30.integrals(IntegralType.exterior_facet)[0],
                    f"tabulate_tensor_{nptype}")
ufcx_form_22, _, _ = jit.ffcx_jit(
    msh.comm, a_22, form_compiler_options={"scalar_type": ffcxtype})
kernel_22 = getattr(ufcx_form_22.integrals(IntegralType.exterior_facet)[0],
                    f"tabulate_tensor_{nptype}")

# Preconditioner form
ufcx_form_p11, _, _ = jit.ffcx_jit(
    msh.comm, p_11, form_compiler_options={"scalar_type": ffcxtype})
kernel_p11 = getattr(ufcx_form_p11.integrals(IntegralType.exterior_facet)[0],
                     f"tabulate_tensor_{nptype}")

# RHS forms
ufcx_form_0, _, _ = jit.ffcx_jit(
    msh.comm, L_0, form_compiler_options={"scalar_type": ffcxtype})
kernel_0 = getattr(ufcx_form_0.integrals(IntegralType.cell)[0],
                   f"tabulate_tensor_{nptype}")

ffi = cffi.FFI()
# FIXME See if there is a better way to pass null
null64 = np.zeros(0, dtype=np.float64)
null32 = np.zeros(0, dtype=np.int32)
null8 = np.zeros(0, dtype=np.uint8)


@numba.njit(fastmath=True)
def compute_mats(coords):
    A_00 = np.zeros((V_ele_space_dim, V_ele_space_dim),
                    dtype=PETSc.ScalarType)
    A_10 = np.zeros((Q_ele_space_dim, V_ele_space_dim),
                    dtype=PETSc.ScalarType)
    A_20 = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                     V_ele_space_dim),
                    dtype=PETSc.ScalarType)
    A_20_f = np.zeros((Vbar_ele_space_dim, V_ele_space_dim),
                      dtype=PETSc.ScalarType)
    A_30 = np.zeros((num_cell_facets * Qbar_ele_space_dim,
                     V_ele_space_dim),
                    dtype=PETSc.ScalarType)
    A_30_f = np.zeros((Qbar_ele_space_dim, V_ele_space_dim),
                      dtype=PETSc.ScalarType)

    A_22 = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                     num_cell_facets * Vbar_ele_space_dim),
                    dtype=PETSc.ScalarType)
    A22_f = np.zeros((Vbar_ele_space_dim, Vbar_ele_space_dim),
                     dtype=PETSc.ScalarType)

    kernel_00_cell(ffi.from_buffer(A_00),
                   ffi.from_buffer(null64),
                   ffi.from_buffer(null64),
                   ffi.from_buffer(coords),
                   ffi.from_buffer(null32),
                   ffi.from_buffer(null8))
    kernel_10(ffi.from_buffer(A_10),
              ffi.from_buffer(null64),
              ffi.from_buffer(null64),
              ffi.from_buffer(coords),
              ffi.from_buffer(null32),
              ffi.from_buffer(null8))

    entity_local_index = np.zeros((1), dtype=np.int32)
    for local_f in range(num_cell_facets):
        entity_local_index[0] = local_f
        A_20_f.fill(0.0)
        A_30_f.fill(0.0)
        A22_f.fill(0.0)

        kernel_00_facet(ffi.from_buffer(A_00),
                        ffi.from_buffer(null64),
                        ffi.from_buffer(null64),
                        ffi.from_buffer(coords),
                        ffi.from_buffer(entity_local_index),
                        ffi.from_buffer(null8))

        kernel_20(ffi.from_buffer(A_20_f),
                  ffi.from_buffer(null64),
                  ffi.from_buffer(null64),
                  ffi.from_buffer(coords),
                  ffi.from_buffer(entity_local_index),
                  ffi.from_buffer(null8))

        kernel_30(ffi.from_buffer(A_30_f),
                  ffi.from_buffer(null64),
                  ffi.from_buffer(null64),
                  ffi.from_buffer(coords),
                  ffi.from_buffer(entity_local_index),
                  ffi.from_buffer(null8))

        kernel_22(ffi.from_buffer(A22_f),
                  ffi.from_buffer(null64),
                  ffi.from_buffer(null64),
                  ffi.from_buffer(coords),
                  ffi.from_buffer(entity_local_index),
                  ffi.from_buffer(null8))

        start_row_20 = local_f * Vbar_ele_space_dim
        end_row_20 = start_row_20 + Vbar_ele_space_dim
        A_20[start_row_20:end_row_20, :] += A_20_f[:, :]

        start_row_30 = local_f * Qbar_ele_space_dim
        end_row_30 = start_row_30 + Qbar_ele_space_dim
        A_30[start_row_30:end_row_30, :] += A_30_f[:, :]

        start = local_f * Vbar_ele_space_dim
        end = start + Vbar_ele_space_dim
        A_22[start:end, start:end] += A22_f[:, :]
    return A_00, A_10, A_20, A_30, A_22


@numba.njit(fastmath=True)
def compute_tilde_mats(A_00, A_10, A_20, A_30):
    # Construct tilde matrices
    A_tilde = np.zeros((V_ele_space_dim + Q_ele_space_dim,
                        V_ele_space_dim + Q_ele_space_dim),
                       dtype=PETSc.ScalarType)
    A_tilde[:V_ele_space_dim, :V_ele_space_dim] = A_00[:, :]
    A_tilde[V_ele_space_dim:, :V_ele_space_dim] = A_10[:, :]
    A_tilde[:V_ele_space_dim, V_ele_space_dim:] = A_10.T[:, :]

    B_tilde = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                        V_ele_space_dim + Q_ele_space_dim),
                       dtype=PETSc.ScalarType)
    B_tilde[:, :V_ele_space_dim] = A_20[:, :]

    C_tilde = np.zeros((num_cell_facets * Qbar_ele_space_dim,
                        V_ele_space_dim + Q_ele_space_dim),
                       dtype=PETSc.ScalarType)
    C_tilde[:, :V_ele_space_dim] = A_30[:, :]

    return A_tilde, B_tilde, C_tilde


@numba.njit(fastmath=True)
def compute_L_tilde(coords):
    b_0 = np.zeros(V_ele_space_dim, dtype=PETSc.ScalarType)
    kernel_0(ffi.from_buffer(b_0),
             ffi.from_buffer(null64),
             ffi.from_buffer(null64),
             ffi.from_buffer(coords),
             ffi.from_buffer(null32),
             ffi.from_buffer(null8))

    L_tilde = np.zeros(V_ele_space_dim + Q_ele_space_dim,
                       dtype=PETSc.ScalarType)
    L_tilde[:V_ele_space_dim] = b_0[:]

    return L_tilde


@numba.njit(fastmath=True)
def numba_print(thing):
    print(thing)


c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_a00(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    A_local = numba.carray(A_, (num_cell_facets * Vbar_ele_space_dim,
                                num_cell_facets * Vbar_ele_space_dim),
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    A_00, A_10, A_20, A_30, A_22 = compute_mats(coords)
    A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)

    A_local += A_22 - B_tilde @ np.linalg.solve(A_tilde, B_tilde.T)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_a01(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    A_local = numba.carray(A_, (num_cell_facets * Vbar_ele_space_dim,
                                num_cell_facets * Qbar_ele_space_dim),
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    A_00, A_10, A_20, A_30, A_22 = compute_mats(coords)
    A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)

    A_local -= B_tilde @ np.linalg.solve(A_tilde, C_tilde.T)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_a10(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    A_local = numba.carray(A_, (num_cell_facets * Qbar_ele_space_dim,
                                num_cell_facets * Vbar_ele_space_dim),
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    A_00, A_10, A_20, A_30, A_22 = compute_mats(coords)
    A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)

    A_local -= C_tilde @ np.linalg.solve(A_tilde, B_tilde.T)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_a11(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    A_local = numba.carray(A_, (num_cell_facets * Qbar_ele_space_dim,
                                num_cell_facets * Qbar_ele_space_dim),
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    A_00, A_10, A_20, A_30, A_22 = compute_mats(coords)
    A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)

    A_local -= C_tilde @ np.linalg.solve(A_tilde, C_tilde.T)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_p00(P_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    P_local = numba.carray(P_, (num_cell_facets * Vbar_ele_space_dim,
                                num_cell_facets * Vbar_ele_space_dim),
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    A_00, A_10, A_20, A_30, A_22 = compute_mats(coords)

    P_local += A_22 - A_20 @ np.linalg.solve(A_00, A_20.T)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_p11(P_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    P_local = numba.carray(P_, (num_cell_facets * Qbar_ele_space_dim,
                                num_cell_facets * Qbar_ele_space_dim),
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    P_11_f = np.zeros((Qbar_ele_space_dim, Qbar_ele_space_dim),
                      dtype=PETSc.ScalarType)

    entity_local_index = np.zeros((1), dtype=np.int32)
    for local_f in range(num_cell_facets):
        entity_local_index[0] = local_f
        P_11_f.fill(0.0)

        kernel_p11(ffi.from_buffer(P_11_f),
                   ffi.from_buffer(null64),
                   ffi.from_buffer(null64),
                   ffi.from_buffer(coords),
                   ffi.from_buffer(entity_local_index),
                   ffi.from_buffer(null8))

        start = local_f * Qbar_ele_space_dim
        end = start + Qbar_ele_space_dim
        P_local[start:end, start:end] += P_11_f[:, :]


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_L0(b_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    b_local = numba.carray(b_, num_cell_facets * Vbar_ele_space_dim,
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)

    A_00, A_10, A_20, A_30, A_22 = compute_mats(coords)
    A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)
    L_tilde = compute_L_tilde(coords)

    b_local -= B_tilde @ np.linalg.solve(A_tilde, L_tilde)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_L1(b_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    b_local = numba.carray(b_, num_cell_facets * Qbar_ele_space_dim,
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)

    A_00, A_10, A_20, A_30, A_22 = compute_mats(coords)
    A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)
    L_tilde = compute_L_tilde(coords)

    b_local -= C_tilde @ np.linalg.solve(A_tilde, L_tilde)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def backsub_u(x_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    x = numba.carray(x_, V_ele_space_dim, dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    w = numba.carray(w_, num_cell_facets * (Vbar_ele_space_dim + Qbar_ele_space_dim),
                     dtype=PETSc.ScalarType)
    u_bar = w[:num_cell_facets * Vbar_ele_space_dim]
    p_bar = w[num_cell_facets * Vbar_ele_space_dim:]

    # FIXME This approach is more expensive then needed. It computes both
    # u and p and then only stores u. Would be better to write backsub
    # expression directly for u.
    A_00, A_10, A_20, A_30, A_22 = compute_mats(coords)
    A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)
    L_tilde = compute_L_tilde(coords)

    U = np.linalg.solve(A_tilde, L_tilde - B_tilde.T @
                        u_bar - C_tilde.T @ p_bar)
    x += U[:V_ele_space_dim]


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def backsub_p(x_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    x = numba.carray(x_, Q_ele_space_dim, dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    w = numba.carray(w_, num_cell_facets * (Vbar_ele_space_dim + Qbar_ele_space_dim),
                     dtype=PETSc.ScalarType)
    u_bar = w[:num_cell_facets * Vbar_ele_space_dim]
    p_bar = w[num_cell_facets * Vbar_ele_space_dim:]

    # FIXME This approach is more expensive then needed. It computes both
    # u and p and then only stores p. Would be better to write backsub
    # expression directly for p.
    A_00, A_10, A_20, A_30, A_22 = compute_mats(coords)
    A_tilde, B_tilde, C_tilde = compute_tilde_mats(A_00, A_10, A_20, A_30)
    L_tilde = compute_L_tilde(coords)

    U = np.linalg.solve(A_tilde, L_tilde - B_tilde.T @
                        u_bar - C_tilde.T @ p_bar)
    x += U[V_ele_space_dim:]


np.set_printoptions(suppress=True, linewidth=200, precision=3)

par_print("Create forms")
integrals_a00 = {
    fem.IntegralType.cell: {-1: (tabulate_tensor_a00.address, [])}}
a00 = Form_float64(
    [Vbar._cpp_object, Vbar._cpp_object], integrals_a00, [], [], False, msh,
    entity_maps={facet_mesh: inv_entity_map})

integrals_a01 = {
    fem.IntegralType.cell: {-1: (tabulate_tensor_a01.address, [])}}
a01 = Form_float64(
    [Vbar._cpp_object, Qbar._cpp_object], integrals_a01, [], [], False, msh,
    entity_maps={facet_mesh: inv_entity_map})

integrals_a10 = {
    fem.IntegralType.cell: {-1: (tabulate_tensor_a10.address, [])}}
a10 = Form_float64(
    [Qbar._cpp_object, Vbar._cpp_object], integrals_a10, [], [], False, msh,
    entity_maps={facet_mesh: inv_entity_map})

integrals_a11 = {
    fem.IntegralType.cell: {-1: (tabulate_tensor_a11.address, [])}}
a11 = Form_float64(
    [Qbar._cpp_object, Qbar._cpp_object], integrals_a11, [], [], False, msh,
    entity_maps={facet_mesh: inv_entity_map})

a = [[a00, a01],
     [a10, a11]]


integrals_p00 = {
    fem.IntegralType.cell: {-1: (tabulate_tensor_p00.address, [])}}
p00 = Form_float64(
    [Vbar._cpp_object, Vbar._cpp_object], integrals_p00, [], [], False, msh,
    entity_maps={facet_mesh: inv_entity_map})

integrals_p11 = {
    fem.IntegralType.cell: {-1: (tabulate_tensor_p11.address, [])}}
p11 = Form_float64(
    [Qbar._cpp_object, Qbar._cpp_object], integrals_p11, [], [], False, msh,
    entity_maps={facet_mesh: inv_entity_map})

p = [[p00, None],
     [None, p11]]

integrals_L0 = {
    fem.IntegralType.cell: {-1: (tabulate_tensor_L0.address, [])}}
L0 = Form_float64(
    [Vbar._cpp_object], integrals_L0, [], [], False, msh,
    entity_maps={facet_mesh: inv_entity_map})

integrals_L1 = {
    fem.IntegralType.cell: {-1: (tabulate_tensor_L1.address, [])}}
L1 = Form_float64(
    [Qbar._cpp_object], integrals_L1, [], [], False, msh,
    entity_maps={facet_mesh: inv_entity_map})

L = [L0, L1]

par_print("Boundary conditions")
msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
facet_mesh_boundary_facets = [inv_entity_map[facet]
                              for facet in msh_boundary_facets]
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
u_bc = fem.Function(Vbar)
u_bc.interpolate(u_e)
bc_ubar = fem.dirichletbc(u_bc, dofs)

pressure_dof = fem.locate_dofs_geometrical(
    Qbar, lambda x: np.logical_and(np.isclose(x[0], 0.0),
                                   np.isclose(x[1], 0.0)))
if len(pressure_dof) > 0:
    pressure_dof = np.array([pressure_dof[0]], dtype=np.int32)

bc_p_bar = fem.dirichletbc(PETSc.ScalarType(0.0), pressure_dof, Qbar)

bcs = [bc_ubar]

use_direct_solver = False
if use_direct_solver:
    bcs.append(bc_p_bar)

par_print("Assemble matrix")
A = assemble_matrix_block_hdg(a, bcs=bcs)
A.assemble()

par_print("Assemble vector")
b = assemble_vector_block_hdg(L, a, bcs=bcs)


if use_direct_solver:
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("superlu_dist")
else:
    par_print("Assemble preconditioner")
    P = assemble_matrix_block_hdg(p, bcs=bcs)
    P.assemble()

    # FIXME Only assemble preconditioner here
    offset_ubar = Vbar.dofmap.index_map.local_range[0] * Vbar.dofmap.index_map_bs + \
        Qbar.dofmap.index_map.local_range[0]
    offset_pbar = offset_ubar + Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs
    is_ubar = PETSc.IS().createStride(Vbar.dofmap.index_map.size_local *
                                      Vbar.dofmap.index_map_bs, offset_ubar, 1, comm=PETSc.COMM_SELF)
    is_pbar = PETSc.IS().createStride(Qbar.dofmap.index_map.size_local,
                                      offset_pbar, 1, comm=PETSc.COMM_SELF)

    null_vec = A.createVecLeft()
    offset = Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs
    null_vec.array[offset:] = 1.0
    null_vec.normalize()
    nsp = PETSc.NullSpace().create(vectors=[null_vec])
    assert nsp.test(A)
    A.setNullSpace(nsp)

    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A, P)
    ksp.setTolerances(rtol=1e-8)
    ksp.setType("minres")
    ksp.getPC().setType("fieldsplit")
    ksp.getPC().setFieldSplitIS(
        ("u", is_ubar),
        ("p", is_pbar))

    # Configure velocity and pressure sub KSPs
    ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
    ksp_u.setType("preonly")
    ksp_u.getPC().setType("hypre")
    ksp_p.setType("preonly")
    ksp_p.getPC().setType("sor")

    # Monitor the convergence of the KSP
    opts = PETSc.Options()
    opts["ksp_monitor"] = None
    opts["ksp_view"] = None
    opts["fieldsplit_u_pc_hypre_type"] = "boomeramg"
    # opts["fieldsplit_u_pc_hypre_boomeramg_cycle_type"] = "V"
    # opts["fieldsplit_u_pc_hypre_boomeramg_grid_sweeps_all"] = 4
    # opts['fieldsplit_u_pc_hypre_boomeramg_strong_threshold'] = 0.5
    # opts["help"] = None
    opts["options_left"] = None
    ksp.setFromOptions()

par_print("Solve")
x = A.createVecRight()
ksp.solve(b, x)

ubar_h = fem.Function(Vbar)
ubar_h.name = "ubar"
pbar_h = fem.Function(Qbar)
pbar_h.name = "pbar"

offset = Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs
ubar_h.x.array[:offset] = x.array_r[:offset]
pbar_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
ubar_h.x.scatter_forward()
pbar_h.x.scatter_forward()

par_print("Write to file")
with io.VTXWriter(msh.comm, "ubar.bp", ubar_h) as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "pbar.bp", pbar_h) as f:
    f.write(0.0)

par_print("Compute error in facet solution")
xbar = ufl.SpatialCoordinate(facet_mesh)
e_ubar = norm_L2(msh.comm, ubar_h - u_e(xbar, ufl))
pbar_h_avg = domain_average(facet_mesh, pbar_h)
pbar_e_avg = domain_average(facet_mesh, p_e(xbar, ufl))
e_pbar = norm_L2(msh.comm, (pbar_h - pbar_h_avg) -
                 (p_e(xbar, ufl) - pbar_e_avg))

par_print("Backsubstitution")
integrals_backsub_u = {fem.IntegralType.cell: {-1: (backsub_u.address, [])}}
u_form = Form_float64([V._cpp_object], integrals_backsub_u,
                      [ubar_h._cpp_object, pbar_h._cpp_object], [], False, None,
                      entity_maps={facet_mesh: inv_entity_map})
coeffs_u = pack_coefficients(u_form)

u_h = fem.Function(V)
fem.assemble_vector(u_h.x.array, u_form, coeffs=coeffs_u)
u_h.vector.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)

integrals_backsub_p = {fem.IntegralType.cell: {-1: (backsub_p.address, [])}}
p_form = Form_float64([Q._cpp_object], integrals_backsub_p,
                      [ubar_h._cpp_object, pbar_h._cpp_object], [], False, None,
                      entity_maps={facet_mesh: inv_entity_map})
coeffs_p = pack_coefficients(p_form)

p_h = fem.Function(Q)
fem.assemble_vector(p_h.x.array, p_form, coeffs=coeffs_p)
p_h.vector.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)

par_print("Write cell fields")
with io.VTXWriter(msh.comm, "u.bp", u_h) as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "p.bp", p_h) as f:
    f.write(0.0)

par_print("Compute erorrs")
x = ufl.SpatialCoordinate(msh)
e_u = norm_L2(msh.comm, u_h - u_e(x, ufl))
e_div_u = norm_L2(msh.comm, div(u_h))
p_h_avg = domain_average(msh, p_h)
p_e_avg = domain_average(msh, p_e(x, ufl))
e_p = norm_L2(msh.comm, (p_h - p_h_avg) - (p_e(x, ufl) - p_e_avg))

num_cells = msh.topology.index_map(tdim).size_global
num_dofs_V = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
num_dofs_Q = Q.dofmap.index_map.size_global
num_dofs_Vbar = Vbar.dofmap.index_map.size_global * Vbar.dofmap.index_map_bs
num_dofs_Qbar = Qbar.dofmap.index_map.size_global
dofs_sc = num_dofs_Vbar + num_dofs_Qbar
total_dofs = num_dofs_V + num_dofs_Q + dofs_sc

if rank == 0:
    print(f"num_cells = {num_cells}")
    print(f"num_dofs_V = {num_dofs_V}")
    print(f"num_dofs_Q = {num_dofs_Q}")
    print(f"num_dofs_Vbar = {num_dofs_Vbar}")
    print(f"num_dofs_Qbar = {num_dofs_Qbar}")
    print(f"dofs_sc = {dofs_sc}")
    print(f"total_dofs = {total_dofs}")
    print(f"e_u = {e_u}")
    print(f"e_div_u = {e_div_u}")
    print(f"e_p = {e_p}")
    print(f"e_ubar = {e_ubar}")
    print(f"e_pbar = {e_pbar}")
