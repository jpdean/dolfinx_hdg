from math import perm
import dolfinx
from dolfinx import UnitSquareMesh, FunctionSpace, Function, DirichletBC
from dolfinx.fem import assemble_scalar
from mpi4py import MPI
from ufl import (TrialFunction, TestFunction, inner, dx, ds, FacetNormal,
                 grad, dot, SpatialCoordinate, sin, pi, div)
from dolfinx_hdg.assemble import assemble_matrix, assemble_vector
import numpy as np
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem import set_bc
from petsc4py import PETSc
from dolfinx_hdg.sc import back_sub
from dolfinx.io import XDMFFile
import numba
import cffi

np.set_printoptions(linewidth=200)

n = 1
mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)

V = FunctionSpace(mesh, ("DG", 1))
# TODO (mesh, codimension=1)
Vbar = FunctionSpace(mesh, ("DG", 1), codimension=1)

V_ele_space_dim = V.dolfin_element().space_dimension()
Vbar_ele_space_dim = Vbar.dolfin_element().space_dimension()

num_facets = mesh.ufl_cell().num_facets()

u = TrialFunction(V)
v = TestFunction(V)
ubar = TrialFunction(Vbar)
vbar = TestFunction(Vbar)

# TODO Use CellDiameter as this will cause recompile.
# FIXME CellDiameter currently not supported by my facet space
# branch
h = 1 / n
gamma = 10.0 / h  # TODO Add dependence on order of polynomials
n = FacetNormal(mesh)

# TODO Check this
a00 = inner(grad(u), grad(v)) * dx - \
    (inner(u, dot(grad(v), n)) * ds + inner(v, dot(grad(u), n)) * ds) + \
    gamma * inner(u, v) * ds
a10 = inner(dot(grad(u), n) - gamma * u, vbar) * ds
a01 = inner(dot(grad(v), n) - gamma * v, ubar) * ds
a11 = gamma * inner(ubar, vbar) * dx

x = SpatialCoordinate(mesh)
u_e = sin(pi * x[0]) * sin(pi * x[1])
f = - div(grad(u_e))

# FIXME Constant doesn't work in my facet space branch
f0 = inner(f, v) * dx
f1 = inner(1e-16, vbar) * dx

# JIT compile individual blocks tabulation kernels
ufc_form00, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), a00)
kernel00_cell = ufc_form00.integrals(0)[0].tabulate_tensor
kernel00_facet = ufc_form00.integrals(1)[0].tabulate_tensor

ufc_form10, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), a10)
kernel10 = ufc_form10.integrals(1)[0].tabulate_tensor

ufc_form11, _, _ = dolfinx.jit.ffcx_jit(mesh.mpi_comm(), a11)
kernel11 = ufc_form11.integrals(0)[0].tabulate_tensor

ffi = cffi.FFI()
c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))

@numba.cfunc(c_signature, nopython=True)
def tabulate_condensed_tensor_A(A_, w_, c_, coords_, entity_local_index,
                                facet_permutations):
    A = numba.carray(A_, (num_facets * Vbar_ele_space_dim,
                          num_facets * Vbar_ele_space_dim),
            dtype=PETSc.ScalarType)
    
    A00 = np.zeros((V_ele_space_dim, V_ele_space_dim), dtype=PETSc.ScalarType)
    # FIXME How do I pass a null pointer for the last two arguments here?
    kernel00_cell(ffi.from_buffer(A00), w_, c_, coords_, entity_local_index,
                  facet_permutations)

    # FIXME Is there a neater way to do this?
    facet = np.zeros((1), dtype=np.int32)
    facet_permutation = np.zeros((1), dtype=np.uint8)
    for i in range(num_facets):
        facet[0] = i
        facet_permutation[0] = facet_permutations[i]
        kernel00_facet(ffi.from_buffer(A00), w_, c_, coords_,
                       ffi.from_buffer(facet), 
                       ffi.from_buffer(facet_permutation))
        A10_f = np.zeros((Vbar_ele_space_dim, V_ele_space_dim),
                          dtype=PETSc.ScalarType)
        kernel10(ffi.from_buffer(A10_f), w_, c_, coords_,
                ffi.from_buffer(facet), 
                ffi.from_buffer(facet_permutation))
        
    A += np.ones_like(A)


integrals = {dolfinx.fem.IntegralType.cell:
             ([(-1, tabulate_condensed_tensor_A.address)], None)}
a = dolfinx.cpp.fem.Form(
    [Vbar._cpp_object, Vbar._cpp_object], integrals, [], [], False, None)
A = assemble_matrix(a)
A.assemble()

np.set_printoptions(linewidth=200, formatter=dict(float=lambda x: "%.3g" % x))
print(A[:, :])


# a = [[a00, a01],
#      [a10, a11]]

# # Boundary conditions
# facets = locate_entities_boundary(mesh, 1,
#                                   lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
#                                                           np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))))
# ubar0 = Function(Vbar)
# dofs_bar = locate_dofs_topological(Vbar, 1, facets)
# bc_bar = DirichletBC(ubar0, dofs_bar)

# A = assemble_matrix(a, [bc_bar])
# A.assemble()

# f = [f0,
#      f1]

# b = assemble_vector(f, a)
# # FIXME apply_lifting not implemented in my facet space branch, so must use homogeneous BC
# set_bc(b, [bc_bar])

# solver = PETSc.KSP().create(mesh.mpi_comm())
# solver.setOperators(A)
# solver.setType("preonly")
# solver.getPC().setType("lu")

# ubar = Function(Vbar)
# solver.solve(b, ubar.vector)

# u = Function(V)
# u.vector[:] = back_sub(ubar.vector, a, f)

# e = u - u_e
# e_L2 = np.sqrt(mesh.mpi_comm().allreduce(
#     assemble_scalar(inner(e, e) * dx), op=MPI.SUM))
# print(f"L2-norm of error = {e_L2}")

# with XDMFFile(MPI.COMM_WORLD, "poisson.xdmf", "w") as file:
#     file.write_mesh(mesh)
#     file.write_function(u)
