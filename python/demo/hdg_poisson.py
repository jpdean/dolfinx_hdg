# FIXME When calling the kernels, I am often passing data that should be
# null

import dolfinx
from dolfinx import UnitSquareMesh, FunctionSpace, Function, DirichletBC
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


# HACK to create facet space
# FIXME Creating the mesh seems to reorder the facets/geomety,
# so facet i in mesh is not the same as facet i in facet_mesh
# FIXME This might be confusing topology and geometry
def create_facet_mesh(mesh):
    x = mesh.geometry.x[:, :-1]
    mesh.topology.create_connectivity(1, 0)
    f_to_v = mesh.topology.connectivity(1, 0)
    facets = f_to_v.array.reshape((-1, 2))

    ufl_cell = ufl.Cell("interval", geometric_dimension=2)
    ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", ufl_cell, 1))
    facet_mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, facets, x, ufl_mesh)

    return facet_mesh


np.set_printoptions(linewidth=200)

print("Set up problem")
n = 1
mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)
facet_mesh = create_facet_mesh(mesh)

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
# # NOTE: This adds the boundary term twice, but it's OK here as we apply
# # homogeneous Dirichlet BCs
a11 = gamma * inner(ubar, vbar) * dx_f

x = SpatialCoordinate(mesh)
u_e = sin(pi * x[0]) * sin(pi * x[1])
f = - div(grad(u_e))

f0 = inner(f, v) * dx_c

print("Compile forms")
# JIT compile individual blocks tabulation kernels
# TODO See if there is an enum for cell and facet integral rather than using
# integer
ufc_form_a00, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), a00)
kernel_a00_cell = ufc_form_a00.integrals(0)[0].tabulate_tensor
kernel_a00_facet = ufc_form_a00.integrals(1)[0].tabulate_tensor

ufc_form_a10, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), a10)
kernel_a10 = ufc_form_a10.integrals(1)[0].tabulate_tensor

ufc_form_a11, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), a11)
kernel_a11 = ufc_form_a11.integrals(0)[0].tabulate_tensor

ufc_form_f0, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), f0)
kernel_f0 = ufc_form_f0.integrals(0)[0].tabulate_tensor

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
def compute_A_blocks(w_, c_, coords_, entity_local_index,
                    facet_permutations):
    A00 = np.zeros((V_ele_space_dim, V_ele_space_dim), dtype=PETSc.ScalarType)
    # FIXME How do I pass a null pointer for the last two arguments here?
    kernel_a00_cell(ffi.from_buffer(A00), w_, c_, coords_, entity_local_index,
                    facet_permutations)
    A10 = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                    V_ele_space_dim), dtype=PETSc.ScalarType)
    A11 = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                    num_cell_facets * Vbar_ele_space_dim),
                    dtype=PETSc.ScalarType)
    coords = numba.carray(coords_,
                          (3 * (num_dofs_g +
                                num_cell_facets * facet_num_dofs_g)),
                          dtype=PETSc.ScalarType)
    cell_coords = coords[:3 * num_dofs_g]

    # FIXME Is there a neater way to do this?
    facet = np.zeros((1), dtype=np.int32)
    facet_permutation = np.zeros((1), dtype=np.uint8)
    for i in range(num_cell_facets):
        facet[0] = i
        facet_permutation[0] = facet_permutations[i]
        kernel_a00_facet(ffi.from_buffer(A00), w_, c_,
                         ffi.from_buffer(cell_coords),
                         ffi.from_buffer(facet),
                         ffi.from_buffer(facet_permutation))
        A10_f = np.zeros((Vbar_ele_space_dim, V_ele_space_dim),
                         dtype=PETSc.ScalarType)
        kernel_a10(ffi.from_buffer(A10_f), w_, c_,
                   ffi.from_buffer(cell_coords),
                   ffi.from_buffer(facet),
                   ffi.from_buffer(facet_permutation))
        A10 += map_A10_f_to_A10(A10_f, i)

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
    return A00, A10, A11


@numba.cfunc(c_signature, nopython=True)
def tabulate_condensed_tensor_A(A_, w_, c_, coords_, entity_local_index,
                                facet_permutations):
    A = numba.carray(A_, (num_cell_facets * Vbar_ele_space_dim,
                          num_cell_facets * Vbar_ele_space_dim),
                     dtype=PETSc.ScalarType)

    A00, A10, A11 = compute_A_blocks(w_, c_, coords_, entity_local_index,
                                     facet_permutations)

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

    A00, A10, A11 = compute_A_blocks(w_, c_, coords_, entity_local_index,
                                     facet_permutations)
    b -= A10 @ np.linalg.solve(A00, b0)


# @numba.cfunc(c_signature, nopython=True)
# def tabulate_x(x_, w_, c_, coords_, entity_local_index,
#                facet_permutations):
#     x = numba.carray(x_, (V_ele_space_dim), dtype=PETSc.ScalarType)
#     xbar = numba.carray(w_, (num_cell_facets * Vbar_ele_space_dim),
#                         dtype=PETSc.ScalarType)
#     # FIXME Don't need to pass w_ here. Pass null instead.
#     # FIXME dolfinx passes nullptr for facetpermutations for a cell
#     # integral. This is a HACK
#     perms = np.zeros((1), dtype=np.uint8)
#     A00, A10 = compute_A00_A10(w_, c_, coords_, entity_local_index,
#                                ffi.from_buffer(perms))
#     b0 = np.zeros((V_ele_space_dim), dtype=PETSc.ScalarType)
#     # FIXME Pass nullptr for last two parameters
#     kernel_f0(ffi.from_buffer(b0), w_, c_, coords_, entity_local_index,
#               facet_permutations)
#     x += np.linalg.solve(A00, b0 - A10.T @ xbar)


print("Boundary conditions")
# Boundary conditions
# FIXME Since mesh and facet mesh don't agree on the facets, check this is
# locating the correct dofs
facets = locate_entities_boundary(mesh, 1,
                                  lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
                                                          np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))))
ubar0 = Function(Vbar)
dofs_bar = locate_dofs_topological(Vbar, 1, facets)
bc_bar = DirichletBC(ubar0, dofs_bar)

print("Assemble LSH")
integrals = {dolfinx.fem.IntegralType.cell:
             ([(-1, tabulate_condensed_tensor_A.address)], None)}
a = dolfinx.cpp.fem.Form(
    [Vbar._cpp_object, Vbar._cpp_object], integrals, [], [], False, mesh)
A = dolfinx_hdg.assemble.assemble_matrix(a, [bc_bar])
A.assemble()

print(A[:, :])

print("Assemble RHS")
integrals = {dolfinx.fem.IntegralType.cell:
             ([(-1, tabulate_condensed_tensor_b.address)], None)}
f = dolfinx.cpp.fem.Form(
    [Vbar._cpp_object], integrals, [], [], False, mesh)
b = dolfinx_hdg.assemble.assemble_vector(f)
# FIXME apply_lifting not implemented in my facet space branch, so must use homogeneous BC
set_bc(b, [bc_bar])

print(b[:])

# print("Solve")
# solver = PETSc.KSP().create(mesh.mpi_comm())
# solver.setOperators(A)
# solver.setType("preonly")
# solver.getPC().setType("lu")

# ubar = Function(Vbar)
# solver.solve(b, ubar.vector)

# print("Pack coefficients")
# packed_ubar = dolfinx_hdg.assemble.pack_facet_space_coeffs_cellwise(ubar, mesh)

# print("Back substitution")
# integrals = {dolfinx.fem.IntegralType.cell:
#              ([(-1, tabulate_x.address)], None)}
# u_form = dolfinx.cpp.fem.Form(
#     [V._cpp_object], integrals, [], [], False, None)

# u = Function(V)
# dolfinx.fem.assemble_vector(u.vector, u_form, coeffs=(None, packed_ubar))

# print("Compute error")
# e = u - u_e
# e_L2 = np.sqrt(mesh.mpi_comm().allreduce(
#     assemble_scalar(inner(e, e) * dx_c), op=MPI.SUM))
# print(f"L2-norm of error = {e_L2}")

# print("Write to file")
# with XDMFFile(MPI.COMM_WORLD, "poisson.xdmf", "w") as file:
#     file.write_mesh(mesh)
#     file.write_function(u)
