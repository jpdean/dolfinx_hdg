from dolfinx import UnitSquareMesh, FunctionSpace, Function, DirichletBC
from mpi4py import MPI
from ufl import (TrialFunction, TestFunction, inner, dx, ds, FacetNormal,
                 grad, dot)
from dolfinx_hdg.assemble import assemble_matrix
import numpy as np
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary


def test_assembly():
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

    # TODO Check
    A_exact = np.array([[1.79537094, 0.76490088, -1.79537094, -0.50870952, 0, 0, -0.26490088, 0.00870952, 0, 0],
                        [0.76490088, 1.78554742, -0.76490088, -0.23509912,
                            0, 0, -1.28554742, -0.26490088, 0, 0],
                        [-1.79537094, -0.76490088, 4.59074187, 2.01741905, -1.79537094, -
                            0.76490088, -0.23509912, -0.50870952, -0.23509912, -0.50870952],
                        [-0.50870952, -0.23509912, 2.01741905, 4.59074187, -0.50870952, -
                            0.23509912, -0.76490088, -1.79537094, -0.76490088, -1.79537094],
                        [0, 0, -1.79537094, -0.50870952, 1.79537094,
                            0.76490088, 0, 0, -0.26490088, 0.00870952],
                        [0, 0, -0.76490088, -0.23509912, 0.76490088,
                            1.78554742, 0, 0, -1.28554742, -0.26490088],
                        [-0.26490088, -1.28554742, -0.23509912, -0.76490088,
                            0, 0, 1.78554742, 0.76490088, 0, 0.],
                        [0.00870952, -0.26490088, -0.50870952, -
                            1.79537094, 0, 0, 0.76490088, 1.79537094, 0, 0],
                        [0, 0, -0.23509912, -0.76490088, -0.26490088, -
                            1.28554742, 0, 0, 1.78554742, 0.76490088],
                        [0, 0, -0.50870952, -1.79537094, 0.00870952, -0.26490088, 0, 0, 0.76490088, 1.79537094]])

    assert(np.allclose(A[:, :], A_exact))

    # Boundary conditions
    facets = locate_entities_boundary(mesh, 1,
                                    lambda x: np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
                                                            np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0))))
    ubar0 = Function(Vbar)
    dofs_bar = locate_dofs_topological(Vbar, 1, facets)
    bc_bar = DirichletBC(ubar0, dofs_bar)

    A = assemble_matrix(a, [bc_bar])
    A.assemble()

    A_exact = np.array([[1, 0, 0,          0,          0, 0, 0, 0, 0, 0],
                        [0, 1, 0,          0,          0, 0, 0, 0, 0, 0],
                        [0, 0, 4.59074187, 2.01741905, 0, 0, 0, 0, 0, 0],
                        [0, 0, 2.01741905, 4.59074187, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0,          0,          1, 0, 0, 0, 0, 0],
                        [0, 0, 0,          0,          0, 1, 0, 0, 0, 0],
                        [0, 0, 0,          0,          0, 0, 1, 0, 0, 0],
                        [0, 0, 0,          0,          0, 0, 0, 1, 0, 0],
                        [0, 0, 0,          0,          0, 0, 0, 0, 1, 0],
                        [0, 0, 0,          0,          0, 0, 0, 0, 0, 1]])

    assert(np.allclose(A[:, :], A_exact))
