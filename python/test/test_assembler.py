import dolfinx
import dolfinx_hdg.assemble
from dolfinx import UnitSquareMesh, FunctionSpace, Function, DirichletBC
from mpi4py import MPI
import numpy as np
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
import numba
from petsc4py import PETSc


def test_assemble_matrix():
    n = 1
    mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)

    Vbar = FunctionSpace(mesh, ("DG", 1), codimension=1)
    Vbar_ele_space_dim = Vbar.dolfin_element().space_dimension()
    num_facets = mesh.ufl_cell().num_facets()

    c_signature = numba.types.void(
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.int32),
        numba.types.CPointer(numba.types.uint8))

    @numba.cfunc(c_signature, nopython=True)
    def tabulate_tensor(A_, w_, c_, coords_, entity_local_index,
                        facet_permutations):
        A = numba.carray(A_, (num_facets * Vbar_ele_space_dim,
                              num_facets * Vbar_ele_space_dim),
                         dtype=PETSc.ScalarType)

        A += np.ones_like(A)

    integrals = {dolfinx.fem.IntegralType.cell:
                 ([(-1, tabulate_tensor.address)], None)}
    a = dolfinx.cpp.fem.Form(
        [Vbar._cpp_object, Vbar._cpp_object], integrals, [], [], False, None)
    A = dolfinx_hdg.assemble.assemble_matrix(a)
    A.assemble()

    A_exact = np.array([[1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 2, 2, 1, 1, 1, 1, 1, 1],
                        [1, 1, 2, 2, 1, 1, 1, 1, 1, 1],
                        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                        [1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                        [1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1],
                        [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]])

    assert(np.allclose(A[:, :], A_exact))

    facets = locate_entities_boundary(mesh, 1,
                                      lambda x: np.logical_or(
                                          np.logical_or(np.isclose(x[0], 0.0),
                                                        np.isclose(x[0], 1.0)),
                                          np.logical_or(np.isclose(x[1], 0.0),
                                                        np.isclose(x[1], 1.0))))
    ubar0 = Function(Vbar)
    dofs = locate_dofs_topological(Vbar, 1, facets)
    bc = DirichletBC(ubar0, dofs)
    A = dolfinx_hdg.assemble.assemble_matrix(a, [bc])
    A.assemble()

    A_exact = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                        [0, 0, 2, 2, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    
    assert(np.allclose(A[:, :], A_exact))


def test_assemble_vector_facet():
    n = 1
    mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)

    Vbar = FunctionSpace(mesh, ("DG", 1), codimension=1)
    Vbar_ele_space_dim = Vbar.dolfin_element().space_dimension()
    num_facets = mesh.ufl_cell().num_facets()

    c_signature = numba.types.void(
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.int32),
        numba.types.CPointer(numba.types.uint8))

    @numba.cfunc(c_signature, nopython=True)
    def tabulate_tensor(b_, w_, c_, coords_, entity_local_index,
                        facet_permutations):
        b = numba.carray(b_, (num_facets * Vbar_ele_space_dim),
                         dtype=PETSc.ScalarType)

        b += np.ones_like(b)

    integrals = {dolfinx.fem.IntegralType.cell:
                 ([(-1, tabulate_tensor.address)], None)}
    f = dolfinx.cpp.fem.Form(
        [Vbar._cpp_object], integrals, [], [], False, None)
    b = dolfinx_hdg.assemble.assemble_vector(f)
    b.assemble()

    b_exact = np.array([1, 1, 2, 2, 1, 1, 1, 1, 1, 1,])

    assert(np.allclose(b[:], b_exact))


def test_assemble_vector_cell():
    n = 1
    mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)

    V = FunctionSpace(mesh, ("DG", 1))
    Vbar = FunctionSpace(mesh, ("DG", 1), codimension=1)
    V_ele_space_dim = V.dolfin_element().space_dimension()

    c_signature = numba.types.void(
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.int32),
        numba.types.CPointer(numba.types.uint8))

    @numba.cfunc(c_signature, nopython=True)
    def tabulate_tensor(b_, w_, c_, coords_, entity_local_index,
                        facet_permutations):
        b = numba.carray(b_, (V_ele_space_dim),
                         dtype=PETSc.ScalarType)

        b += np.ones_like(b)

    integrals = {dolfinx.fem.IntegralType.cell:
                 ([(-1, tabulate_tensor.address)], None)}
    f = dolfinx.cpp.fem.Form(
        [V._cpp_object], integrals, [], [], False, None)
    b = dolfinx_hdg.assemble.assemble_vector(f)
    b.assemble()

    b_exact = np.array([1, 1, 1, 1, 1, 1])

    assert(np.allclose(b[:], b_exact))


def test_coeff():
    n = 1
    mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)

    Vbar = FunctionSpace(mesh, ("DG", 1), codimension=1)
    Vbar_ele_space_dim = Vbar.dolfin_element().space_dimension()
    num_facets = mesh.ufl_cell().num_facets()

    xbar = Function(Vbar)
    xbar.vector.set(1)

    c_signature = numba.types.void(
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.int32),
        numba.types.CPointer(numba.types.uint8))

    @numba.cfunc(c_signature, nopython=True)
    def tabulate_tensor(b_, w_, c_, coords_, entity_local_index,
                        facet_permutations):
        b = numba.carray(b_, (num_facets * Vbar_ele_space_dim),
                         dtype=PETSc.ScalarType)
        xbar = numba.carray(w_, (num_facets * Vbar_ele_space_dim),
                            dtype=PETSc.ScalarType)
        b += xbar

    integrals = {dolfinx.fem.IntegralType.cell:
                 ([(-1, tabulate_tensor.address)], None)}
    f = dolfinx.cpp.fem.Form(
        [Vbar._cpp_object], integrals, [xbar._cpp_object], [], False, None)
    b = dolfinx_hdg.assemble.assemble_vector(f)
    b.assemble()

    b_exact = np.array([1, 1, 2, 2, 1, 1, 1, 1, 1, 1,])

    assert(np.allclose(b[:], b_exact))
