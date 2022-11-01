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
out_str = f"rank {rank}:\n"

n = 8
msh = mesh.create_unit_square(
    comm, n, n, ghost_mode=mesh.GhostMode.none)

# Currently, permutations are not working in parallel, so reorder the
# mesh
reorder_mesh(msh)

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

k = 2
V = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k))
Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k - 1))
Vbar = fem.VectorFunctionSpace(
    facet_mesh, ("Discontinuous Lagrange", k))
Qbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

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
gamma = 6.0 * k**2 / h


def u_e(x):
    return ufl.as_vector(
        (x[0]**2 * (1 - x[0])**2 * (2 * x[1] - 6 * x[1]**2 + 4 * x[1]**3),
         - x[1]**2 * (1 - x[1])**2 * (2 * x[0] - 6 * x[0]**2 + 4 * x[0]**3)))


def p_e(x):
    return x[0] * (1 - x[0])


dx_c = ufl.Measure("dx", domain=msh)
ds_c = ufl.Measure("ds", domain=msh)

x = ufl.SpatialCoordinate(msh)
f = - div(grad(u_e(x))) + grad(p_e(x))

a_00 = inner(grad(u), grad(v)) * dx_c + gamma * inner(u, v) * ds_c \
    - (inner(u, dot(grad(v), n))
       + inner(v, dot(grad(u), n))) * ds_c
a_10 = - inner(q, div(u)) * dx_c
a_20 = inner(vbar, dot(grad(u), n)) * ds_c - gamma * inner(vbar, u) * ds_c
a_30 = inner(dot(u, n), qbar) * ds_c
a_22 = gamma * inner(ubar, vbar) * ds_c

L_0 = inner(f, v) * dx_c

inv_entity_map = np.full_like(entity_map, -1)
for i, f in enumerate(entity_map):
    inv_entity_map[f] = i

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

    return A_tilde, B_tilde, C_tilde, A_22


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
    A_tilde, B_tilde, C_tilde, A_22 = compute_mats(coords)

    A_local += A_22 - B_tilde @ np.linalg.solve(A_tilde, B_tilde.T)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_a01(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    A_local = numba.carray(A_, (num_cell_facets * Vbar_ele_space_dim,
                                num_cell_facets * Qbar_ele_space_dim),
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    A_tilde, B_tilde, C_tilde, A_22 = compute_mats(coords)

    A_local -= B_tilde @ np.linalg.solve(A_tilde, C_tilde.T)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_a10(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    A_local = numba.carray(A_, (num_cell_facets * Qbar_ele_space_dim,
                                num_cell_facets * Vbar_ele_space_dim),
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    A_tilde, B_tilde, C_tilde, A_22 = compute_mats(coords)

    A_local -= C_tilde @ np.linalg.solve(A_tilde, B_tilde.T)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_a11(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    A_local = numba.carray(A_, (num_cell_facets * Qbar_ele_space_dim,
                                num_cell_facets * Qbar_ele_space_dim),
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
    A_tilde, B_tilde, C_tilde, A_22 = compute_mats(coords)

    A_local -= C_tilde @ np.linalg.solve(A_tilde, C_tilde.T)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_L0(b_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    b_local = numba.carray(b_, num_cell_facets * Vbar_ele_space_dim,
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
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

    A_tilde, B_tilde, C_tilde, A_22 = compute_mats(coords)
    b_local -= B_tilde @ np.linalg.solve(A_tilde, L_tilde)


@numba.cfunc(c_signature, nopython=True, fastmath=True)
def tabulate_tensor_L1(b_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
    b_local = numba.carray(b_, num_cell_facets * Qbar_ele_space_dim,
                           dtype=PETSc.ScalarType)
    coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)

    # FIXME This is duplicated in tabulate_tensor_L0
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

    A_tilde, B_tilde, C_tilde, A_22 = compute_mats(coords)

    b_local -= C_tilde @ np.linalg.solve(A_tilde, L_tilde)


np.set_printoptions(suppress=True, linewidth=200, precision=3)

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

msh_boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary)
facet_mesh_boundary_facets = [inv_entity_map[facet]
                              for facet in msh_boundary_facets]
dofs = fem.locate_dofs_topological(Vbar, fdim, facet_mesh_boundary_facets)
bc_ubar = fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), dofs, Vbar)

pressure_dof = fem.locate_dofs_geometrical(
    Qbar, lambda x: np.logical_and(np.isclose(x[0], 0.0),
                                   np.isclose(x[1], 0.0)))
if len(pressure_dof) > 0:
    pressure_dof = np.array([pressure_dof[0]], dtype=np.int32)

bc_p = fem.dirichletbc(PETSc.ScalarType(0.0), pressure_dof, Qbar)

bcs = [bc_ubar, bc_p]

A = assemble_matrix_block_hdg(a, bcs=bcs)
A.assemble()
print(A.norm())

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

b = assemble_vector_block_hdg(L, a, bcs=bcs)

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("superlu_dist")

x = A.createVecRight()
ksp.solve(b, x)

ubar_h = fem.Function(Vbar)
ubar_h.name = "ubar"
pbar_h = fem.Function(Qbar)
pbar_h.name = "pbar"

offset = Vbar.dofmap.index_map.size_local * Vbar.dofmap.index_map_bs
ubar_h.x.array[:offset] = x.array_r[:offset]
pbar_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]

with io.VTXWriter(msh.comm, "ubar.bp", ubar_h) as f:
    f.write(0.0)
with io.VTXWriter(msh.comm, "pbar.bp", pbar_h) as f:
    f.write(0.0)

xbar = ufl.SpatialCoordinate(facet_mesh)
e_ubar = norm_L2(msh.comm, ubar_h - u_e(xbar))
pbar_h_avg = domain_average(facet_mesh, pbar_h)
pbar_e_avg = domain_average(facet_mesh, p_e(xbar))
e_pbar = norm_L2(msh.comm, (pbar_h - pbar_h_avg) - (p_e(xbar) - pbar_e_avg))

if rank == 0:
    # print(f"e_u = {e_u}")
    # print(f"e_div_u = {e_div_u}")
    # print(f"e_p = {e_p}")
    print(f"e_ubar = {e_ubar}")
    print(f"e_pbar = {e_pbar}")
