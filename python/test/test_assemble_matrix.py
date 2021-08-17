from dolfinx import UnitSquareMesh, FunctionSpace
from mpi4py import MPI
from ufl import TrialFunction, TestFunction, inner, dx, ds
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

a00 = inner(u, v) * dx + inner(u, v) * ds
a10 = inner(u, vbar) * ds
a01 = inner(v, ubar) * ds
a11 = inner(ubar, vbar) * dx

a = [[a00, a01],
     [a10, a11]]

A = assemble_matrix(a)
A.assemble()

print(A[:, :])
