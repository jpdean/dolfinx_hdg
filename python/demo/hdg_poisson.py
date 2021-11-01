# FIXME When calling the kernels, I am often passing data that should be
# null
# TODO Go over code now that I'm using a facet mesh and see if it can be simplified

import dolfinx
from dolfinx import UnitSquareMesh, UnitCubeMesh, FunctionSpace, Function, DirichletBC
from dolfinx.fem import assemble_scalar
from mpi4py import MPI
from ufl import (TrialFunction, TestFunction, inner, FacetNormal,
                 grad, dot, SpatialCoordinate, sin, pi, div)
import dolfinx_hdg.assemble
import dolfinx
import numpy as np
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import set_bc
from petsc4py import PETSc
from dolfinx.io import XDMFFile
import numba
import cffi
import ufl
import random


def create_random_mesh(N):
    random.seed(6)
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    temp_points = np.array([[x / 2, y / 2] for y in range(N) for x in range(N)])
    order = [i for i, j in enumerate(temp_points)]
    # random.shuffle(order)
    points = np.zeros(temp_points.shape)
    for i, j in enumerate(order):
        points[j] = temp_points[i]
    cells = []
    for x in range(N - 1):
        for y in range(N - 1):
            a = N * y + x
            # Adds two triangle cells:
            # a+N -- a+N+1
            #  |   / |
            #  |  /  |
            #  | /   |
            #  a --- a+1
            for cell in [[a, a + 1, a + N + 1], [a, a + N + 1, a + N]]:
                cells.append([order[i] for i in cell])
    points /= np.max(points)
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, points, domain)


def create_custom_mesh(ordered):
    domain = ufl.Mesh(ufl.VectorElement("Lagrange", "triangle", 1))
    x = np.array([[0.5, 0. ],
                  [1. , 0. ],
                  [1. , 0.5],
                  [0.5, 0.5],
                  [0. , 0. ],
                  [1. , 1. ],
                  [0. , 0.5],
                  [0.5, 1. ],
                  [0. , 1. ]])
    if ordered:
        cells = [[0, 1, 2],
                 [0, 3, 2],
                 [4, 0, 3],
                 [3, 2, 5],
                 [4, 6, 3],
                 [3, 7, 5],
                 [6, 3, 7],
                 [6, 8, 7]]
    else:
        cells = [[0, 1, 2],
                 [0, 3, 2],
                 [4, 3, 0],
                 [3, 2, 5],
                 [4, 6, 3],
                 [3, 7, 5],
                 [6, 3, 7],
                 [6, 8, 7]]
    mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, domain)
    return mesh


# Creating a facet mesh causes warnings about the same facet being
# in more than two cells, so disable warning logs
dolfinx.cpp.log.set_log_level(dolfinx.cpp.log.LogLevel.ERROR)

np.set_printoptions(linewidth=200)

print("Set up problem")
# FIME n is not the same for both meshes
# ordered = False
# mesh = create_custom_mesh(ordered)
n = 1
# mesh = create_random_mesh(n + 1)
mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)
# mesh = UnitCubeMesh(MPI.COMM_WORLD, n, n, n)
facet_dim = mesh.topology.dim - 1
# FIXME What is the best way to get the total number of facets?
mesh.topology.create_connectivity(facet_dim, 0)
num_mesh_facets = mesh.topology.connectivity(facet_dim, 0).num_nodes
# print(num_mesh_facets)
# mesh.topology.create_entity_permutations()
# facet_perms = mesh.topology.get_facet_permutations()
# print(facet_perms)
# print(np.where(facet_perms))
facets = np.arange(num_mesh_facets, dtype=np.int32)
facet_mesh = mesh.sub(facet_dim, facets)

k = 1
V = FunctionSpace(mesh, ("DG", k))
Vbar = FunctionSpace(facet_mesh, ("DG", k))

V_ele_space_dim = V.dolfin_element().space_dimension()
Vbar_ele_space_dim = Vbar.dolfin_element().space_dimension()

num_cell_facets = mesh.ufl_cell().num_facets()
num_dofs_g = len(mesh.geometry.dofmap.links(0))
facet_num_dofs_g = len(facet_mesh.geometry.dofmap.links(0))

u = TrialFunction(V)
v = TestFunction(V)
ubar = TrialFunction(Vbar)
vbar = TestFunction(Vbar)

# # TODO Use CellDiameter as this will cause recompile.
# # FIXME CellDiameter currently not supported by my facet space
# # branch
h = 1 / n
gamma = 10.0 * k**2 / h
n = FacetNormal(mesh)

dx_c = ufl.Measure("dx", domain=mesh)
ds_c = ufl.Measure("ds", domain=mesh)
dx_f = ufl.Measure("dx", domain=facet_mesh)

a00 = inner(grad(u), grad(v)) * dx_c - \
    (inner(u, dot(grad(v), n)) * ds_c + inner(v, dot(grad(u), n)) * ds_c) + \
    gamma * inner(u, v) * ds_c
a10 = inner(dot(grad(u), n) - gamma * u, vbar) * ds_c
a11 = gamma * inner(ubar, vbar) * dx_f

x = SpatialCoordinate(mesh)
tdim = mesh.topology.dim
u_e = 1
for i in range(tdim):
    u_e *= sin(pi * x[i])
f = - div(grad(u_e))

f0 = inner(f, v) * dx_c

print("Compile forms")
# JIT compile individual blocks tabulation kernels
# TODO See if there is an enum for cell and facet integral rather than using
# integer
nptype = "float64"
ffcxtype = "double"
form_compiler_parameters={"scalar_type": ffcxtype}

ufc_form_a00, _, _ = dolfinx.jit.ffcx_jit(
    mesh.mpi_comm(), a00, form_compiler_parameters=form_compiler_parameters)
kernel_a00_cell = getattr(ufc_form_a00.integrals(0)[0], f"tabulate_tensor_{nptype}")
kernel_a00_facet = getattr(ufc_form_a00.integrals(1)[0], f"tabulate_tensor_{nptype}")

ufc_form_a10, _, _ = dolfinx.jit.ffcx_jit(
    mesh.mpi_comm(), a10, form_compiler_parameters=form_compiler_parameters)
kernel_a10 = getattr(ufc_form_a10.integrals(1)[0], f"tabulate_tensor_{nptype}")

ufc_form_a11, _, _ = dolfinx.jit.ffcx_jit(
    mesh.mpi_comm(), a11, form_compiler_parameters=form_compiler_parameters)
kernel_a11 = getattr(ufc_form_a11.integrals(0)[0], f"tabulate_tensor_{nptype}")

ufc_form_f0, _, _ = dolfinx.jit.ffcx_jit(
    mesh.mpi_comm(), f0, form_compiler_parameters=form_compiler_parameters)
kernel_f0 = getattr(ufc_form_f0.integrals(0)[0], f"tabulate_tensor_{nptype}")

print("Compile numba")
ffi = cffi.FFI()
c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))


@numba.jit(nopython=True)
def map_A10_f_to_A10(A10_f, f):
    A = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                  V_ele_space_dim), dtype=PETSc.ScalarType)
    start_row = f * Vbar_ele_space_dim
    end_row = f * Vbar_ele_space_dim + Vbar_ele_space_dim
    A[start_row:end_row, :] += A10_f[:, :]
    return A


@numba.jit(nopython=True)
def map_A11_f_to_A11(A11_f, f):
    A = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                  num_cell_facets * Vbar_ele_space_dim,),
                 dtype=PETSc.ScalarType)
    start = f * Vbar_ele_space_dim
    end = start + Vbar_ele_space_dim
    A[start:end, start:end] += A11_f[:, :]
    return A


@numba.jit(nopython=True)
def compute_A00_A10(w_, c_, coords_, entity_local_index,
                    facet_permutations):
    A00 = np.zeros((V_ele_space_dim, V_ele_space_dim), dtype=PETSc.ScalarType)
    # FIXME How do I pass a null pointer for the last two arguments here?
    kernel_a00_cell(ffi.from_buffer(A00), w_, c_, coords_, entity_local_index,
                    facet_permutations)
    A10 = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                    V_ele_space_dim), dtype=PETSc.ScalarType)

    # FIXME Is there a neater way to do this?
    facet = np.zeros((1), dtype=np.int32)
    facet_permutation = np.zeros((1), dtype=np.uint8)
    for i in range(num_cell_facets):
        facet[0] = i
        facet_permutation[0] = facet_permutations[i]
        kernel_a00_facet(ffi.from_buffer(A00), w_, c_, coords_,
                         ffi.from_buffer(facet),
                         ffi.from_buffer(facet_permutation))
        A10_f = np.zeros((Vbar_ele_space_dim, V_ele_space_dim),
                         dtype=PETSc.ScalarType)
        kernel_a10(ffi.from_buffer(A10_f), w_, c_, coords_,
                   ffi.from_buffer(facet),
                   ffi.from_buffer(facet_permutation))
        A10 += map_A10_f_to_A10(A10_f, i)
    # print("A00")
    # print(A00)
    # print("A10")
    # print(A10)
    # FIXME HACK to permute dofs by flipping. Figure out proper way to do
    # this
    for i in range(num_cell_facets):
        facet_permutation[0] = facet_permutations[i]
        if (facet_permutation == 1):
            r_0 = np.copy(A10[2 * i, :])
            r_1 = np.copy(A10[2 * i + 1, :])
            A10[2 * i, :] = r_1
            A10[2 * i + 1, :] = r_0
    return A00, A10


@numba.jit(nopython=True)
def compute_A11(w_, c_, coords_, entity_local_index,
                facet_permutations):
    # FIXME How do I pass a null pointer for the last two arguments here?
    A11 = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                    num_cell_facets * Vbar_ele_space_dim),
                    dtype=PETSc.ScalarType)
    coords = numba.carray(coords_,
                          (3 * (num_dofs_g +
                                num_cell_facets * facet_num_dofs_g)),
                          dtype=PETSc.ScalarType)

    for i in range(num_cell_facets):
        offset = 3 * (num_dofs_g + i * facet_num_dofs_g)
        facet_coords = coords[offset:offset + 3 * facet_num_dofs_g]
        A11_f = np.zeros((Vbar_ele_space_dim, Vbar_ele_space_dim),
                         dtype=PETSc.ScalarType)
        # FIXME Last two should be nullptr
        kernel_a11(ffi.from_buffer(A11_f), w_, c_,
                   ffi.from_buffer(facet_coords),
                   entity_local_index,
                   facet_permutations)
        A11 += map_A11_f_to_A11(A11_f, i)
    
    # print("A11")
    # print(A11)
    return A11


@numba.cfunc(c_signature, nopython=True)
def tabulate_condensed_tensor_A(A_, w_, c_, coords_, entity_local_index,
                                facet_permutations):
    A = numba.carray(A_, (num_cell_facets * Vbar_ele_space_dim,
                          num_cell_facets * Vbar_ele_space_dim),
                     dtype=PETSc.ScalarType)

    A00, A10 = compute_A00_A10(w_, c_, coords_, entity_local_index,
                                     facet_permutations)
    A11 = compute_A11(w_, c_, coords_, entity_local_index, facet_permutations)

    A += A11 - A10 @ np.linalg.solve(A00, A10.T)


@numba.cfunc(c_signature, nopython=True)
def tabulate_condensed_tensor_b(b_, w_, c_, coords_, entity_local_index,
                                facet_permutations):
    b = numba.carray(b_, (num_cell_facets * Vbar_ele_space_dim),
                     dtype=PETSc.ScalarType)

    b0 = np.zeros((V_ele_space_dim), dtype=PETSc.ScalarType)
    # TODO Pass nullptr for last two parameters
    kernel_f0(ffi.from_buffer(b0), w_, c_, coords_, entity_local_index,
              facet_permutations)

    A00, A10 = compute_A00_A10(w_, c_, coords_, entity_local_index,
                               facet_permutations)
    b -= A10 @ np.linalg.solve(A00, b0)


@numba.cfunc(c_signature, nopython=True)
def tabulate_x(x_, w_, c_, coords_, entity_local_index,
               facet_permutations):
    x = numba.carray(x_, (V_ele_space_dim), dtype=PETSc.ScalarType)
    xbar = numba.carray(w_, (num_cell_facets * Vbar_ele_space_dim),
                        dtype=PETSc.ScalarType)
    # FIXME Don't need to pass w_ here. Pass null instead.
    # FIXME dolfinx passes nullptr for facetpermutations for a cell
    # integral. This is a HACK
    perms = np.zeros((1), dtype=np.uint8)
    A00, A10 = compute_A00_A10(w_, c_, coords_, entity_local_index,
                               ffi.from_buffer(perms))
    b0 = np.zeros((V_ele_space_dim), dtype=PETSc.ScalarType)
    # FIXME Pass nullptr for last two parameters
    kernel_f0(ffi.from_buffer(b0), w_, c_, coords_, entity_local_index,
              facet_permutations)
    x += np.linalg.solve(A00, b0 - A10.T @ xbar)


# Boundary conditions
print("Boundary conditions")


def boundary(x):
    lr = np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    tb = np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))
    lrtb = np.logical_or(lr, tb)
    if tdim == 2:
        return lrtb
    else:
        fb = np.logical_or(np.isclose(x[2], 0.0), np.isclose(x[2], 1.0))
        return np.logical_or(lrtb, fb)


# FIXME Since mesh and facet mesh don't agree on the facets, check this is
# locating the correct dofs
facets = locate_entities_boundary(mesh, tdim - 1, boundary)
ubar0 = Function(Vbar)
dofs_bar = locate_dofs_topological(Vbar, tdim - 1, facets)
bc_bar = DirichletBC(ubar0, dofs_bar)

print("Assemble LSH")
Form = dolfinx.cpp.fem.Form_float64
integrals = {dolfinx.fem.IntegralType.cell:
             ([(-1, tabulate_condensed_tensor_A.address)], None)}
# NOTE I've disabled facet perms
a = Form([Vbar._cpp_object, Vbar._cpp_object], integrals, [], [], True, mesh)
A = dolfinx_hdg.assemble.assemble_matrix(a, [bc_bar])
A.assemble()

# A = A[:, :]

# if ordered:
#     A_f_name = "A_ordered.txt"
# else:
#     A_f_name = "A_unordered.txt"

# np.savetxt(A_f_name, A)

# b = np.ones(A.shape[0])

# xbar = np.linalg.solve(A, b)
# ubar = Function(Vbar)
# ubar.vector[:] = xbar[:]


# print(A[:, :])
# print(np.unique(np.round(A, 6)))
# np.savetxt("A_ordered.txt", A[:, :])
# print(mesh.topology.)
# print(np.where(np.isclose(A[:, :], 3.590742)))
# print(np.where(np.isclose(A[:, :], 3.571095)))

# with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as file:
#     file.write_mesh(mesh)

print("Assemble RHS")
integrals = {dolfinx.fem.IntegralType.cell:
             ([(-1, tabulate_condensed_tensor_b.address)], None)}
f = Form([Vbar._cpp_object], integrals, [], [], True, mesh)
b = dolfinx_hdg.assemble.assemble_vector(f)
# FIXME apply_lifting not implemented in my facet space branch, so must use homogeneous BC
set_bc(b, [bc_bar])

print("Solve")
solver = PETSc.KSP().create(mesh.mpi_comm())
solver.setOperators(A)
solver.setType("preonly")
solver.getPC().setType("lu")

ubar = Function(Vbar)
solver.solve(b, ubar.vector)

with XDMFFile(MPI.COMM_WORLD, "ubar.xdmf", "w") as file:
    file.write_mesh(facet_mesh)
    file.write_function(ubar)

print("Pack coefficients")
# FIXME Temporary hack until we have reworked pack coefficients. Should pass
# coefficient to form and pack properly there
packed_ubar = dolfinx_hdg.assemble.pack_facet_space_coeffs_cellwise(ubar, mesh)

print("Back substitution")
integrals = {dolfinx.fem.IntegralType.cell:
             ([(-1, tabulate_x.address)], None)}
u_form = Form([V._cpp_object], integrals, [], [], True, None)

u = Function(V)
dolfinx_hdg.assemble.assemble_vector(u.vector, u_form, coeffs=(None, packed_ubar))

# print("Compute error")
# e = u - u_e
# e_L2 = np.sqrt(mesh.mpi_comm().allreduce(
#     assemble_scalar(inner(e, e) * dx_c), op=MPI.SUM))
# print(f"L2-norm of error = {e_L2}")

# print("Write to file")
# with XDMFFile(MPI.COMM_WORLD, "poisson_u.xdmf", "w") as file:
#     file.write_mesh(mesh)
#     file.write_function(u)

# with XDMFFile(MPI.COMM_WORLD, "poisson_ubar.xdmf", "w") as file:
#     file.write_mesh(facet_mesh)
#     file.write_function(ubar)

# print("Done")
