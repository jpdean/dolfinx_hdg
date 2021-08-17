from dolfinx import UnitSquareMesh, FunctionSpace
from mpi4py import MPI
from ufl import (TrialFunction, TestFunction, inner, dx, ds, FacetNormal,
                 grad, dot)
from dolfinx_hdg.assemble import assemble_matrix
import numpy as np

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

a = [[a00, a01],
     [a10, a11]]

A = assemble_matrix(a)
A.assemble()

# NOTE If incorrect, it could be due to not applying transformations.
# Check if appy transformations does anything in facet space assemble
print(A[:, :])
