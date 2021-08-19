from dolfinx import UnitSquareMesh, FunctionSpace, Function, DirichletBC
from mpi4py import MPI
from ufl import (TrialFunction, TestFunction, inner, dx, ds, FacetNormal,
                 grad, dot)
from dolfinx_hdg.assemble import assemble_matrix, assemble_vector
import numpy as np
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary

np.set_printoptions(linewidth=200)

n = 1
mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)

V = FunctionSpace(mesh, ("DG", 1))
Vbar = FunctionSpace(mesh, ("DG", 1), codimension=1)

u = TrialFunction(V)
v = TestFunction(V)
ubar = TrialFunction(Vbar)
vbar = TestFunction(Vbar)

h = 1 / n  # TODO Use CellDiameter
gamma = 10.0 / h  # TODO Add dependence on order of polynomials
n = FacetNormal(mesh)

# TODO Check this
a00 = inner(grad(u), grad(v)) * dx - \
    (inner(u, dot(grad(v), n)) * ds + inner(v, dot(grad(u), n)) * ds) + \
    gamma * inner(u, v) * ds
a10 = inner(dot(grad(u), n) - gamma * u, vbar) * ds
a01 = inner(dot(grad(v), n) - gamma * v, ubar) * ds
a11 = gamma * inner(ubar, vbar) * dx
# FIXME Constant doesn't work in my facet space branch
f0 = inner(1, v) * dx
f1 = inner(1e-16, vbar) * dx

a = [[a00, a01],
     [a10, a11]]

# Boundary conditions
facets = locate_entities_boundary(mesh, 1,
                                  lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
                                                          np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))))
ubar0 = Function(Vbar)
dofs_bar = locate_dofs_topological(Vbar, 1, facets)
bc_bar = DirichletBC(ubar0, dofs_bar)

A = assemble_matrix(a, [bc_bar])
A.assemble()
# print(A[:, :])

f = [f0,
     f1]

b = assemble_vector(f, a)
print(b[:])
