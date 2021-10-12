import dolfinx
import dolfinx_hdg.assemble
from dolfinx import UnitSquareMesh, UnitCubeMesh, FunctionSpace, Function, DirichletBC
from mpi4py import MPI
import numpy as np
from dolfinx.fem import locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
import numba
from petsc4py import PETSc
import ufl
import pytest
from dolfinx.fem import set_bc


# HACK to create facet space
# FIXME Creating the mesh seems to reorder the facets/geomety,
# so facet i in mesh is not the same as facet i in facet_mesh
# FIXME This might be confusing topology and geometry
def create_facet_mesh(mesh):
    tdim = mesh.topology.dim

    if tdim == 2:
        x = mesh.geometry.x[:, :-1]
    else:
        assert(tdim == 3)
        x = mesh.geometry.x
    mesh.topology.create_connectivity(tdim - 1, 0)
    f_to_v = mesh.topology.connectivity(tdim - 1, 0)

    # HACK This only works because in d-dimensional simplices have d vertices
    facets = f_to_v.array.reshape((-1, tdim))

    facet_cell = mesh.ufl_cell().facet_cell()
    ufl_mesh = ufl.Mesh(ufl.VectorElement("Lagrange", facet_cell, 1))
    facet_mesh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, facets, x, ufl_mesh)

    return facet_mesh


@pytest.mark.parametrize("d", [2, 3])
def test_assemble_matrix(d):
    n = 2
    if d == 2:
        mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)
    else:
        assert(d == 3)
        mesh = UnitCubeMesh(MPI.COMM_WORLD, n, n, n)

    # HACK Create a second mesh where the "facets"
    facet_mesh = create_facet_mesh(mesh)

    Vbar = dolfinx.FunctionSpace(facet_mesh, ("DG", 1))
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
        [Vbar._cpp_object, Vbar._cpp_object], integrals, [], [], False, mesh)
    A = dolfinx_hdg.assemble.assemble_matrix(a)
    A.assemble()

    A_expected = np.zeros_like(A[:, :])
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim - 1)
    c_to_f = mesh.topology.connectivity(tdim, tdim -1)
    for cell in range(c_to_f.num_nodes):
        facets = c_to_f.links(cell)
        for facet_i in facets:
            for facet_j in facets:
                dofs_i = Vbar.dofmap.list.links(facet_i)
                dofs_j = Vbar.dofmap.list.links(facet_j)
                # FIXME Use numpy slices rather than so many
                # for loops
                for dof_i in dofs_i:
                    for dof_j in dofs_j:
                        A_expected[dof_i, dof_j] += 1
                
    assert(np.allclose(A[:, :], A_expected))

    facets = locate_entities_boundary(mesh, tdim - 1,
                                      lambda x: np.logical_or(
                                          np.logical_or(np.isclose(x[0], 0.0),
                                                        np.isclose(x[0], 1.0)),
                                          np.logical_or(np.isclose(x[1], 0.0),
                                                        np.isclose(x[1], 1.0))))
    dofs = locate_dofs_topological(Vbar, tdim - 1, facets)
    ubar0 = Function(Vbar)
    bc = DirichletBC(ubar0, dofs)
    A = dolfinx_hdg.assemble.assemble_matrix(a, [bc])
    A.assemble()

    for dof in dofs:
        A_expected[dof, :] = 0
        A_expected[:, dof] = 0
        A_expected[dof, dof] = 1
    
    assert(np.allclose(A[:, :], A_expected))


def test_assemble_vector():
    n = 2
    mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)

    # HACK Create a second mesh where the "facets"
    facet_mesh = create_facet_mesh(mesh)

    Vbar = dolfinx.FunctionSpace(facet_mesh, ("DG", 1))
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
        [Vbar._cpp_object], integrals, [], [], False, mesh)
    b = dolfinx_hdg.assemble.assemble_vector(f)
    b.assemble()

    b_expected = np.zeros_like(b[:])
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim - 1)
    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
    for cell in range(c_to_f.num_nodes):
        facets = c_to_f.links(cell)
        for facet in facets:
            dofs = Vbar.dofmap.list.links(facet)
            # FIXME Use numpy slices rather than so many
            # for loops
            for dof in dofs:
                b_expected[dof] += 1

    assert(np.allclose(b[:], b_expected))

    facets = locate_entities_boundary(mesh, tdim - 1,
                                      lambda x: np.logical_or(
                                          np.logical_or(np.isclose(x[0], 0.0),
                                                        np.isclose(x[0], 1.0)),
                                          np.logical_or(np.isclose(x[1], 0.0),
                                                        np.isclose(x[1], 1.0))))
    dofs = locate_dofs_topological(Vbar, tdim - 1, facets)
    ubar0 = Function(Vbar)
    bc = DirichletBC(ubar0, dofs)
    set_bc(b, [bc])
    
    for dof in dofs:
        b_expected[dof] = 0
    
    assert(np.allclose(b[:], b_expected))


def test_backsub():
    n = 1
    mesh = UnitSquareMesh(MPI.COMM_WORLD, n, n)

    # HACK Create a second mesh where the "facets"
    facet_mesh = create_facet_mesh(mesh)

    V =dolfinx.FunctionSpace(mesh, ("DG", 1))
    Vbar = dolfinx.FunctionSpace(facet_mesh, ("DG", 1))
    V_ele_space_dim = V.dolfin_element().space_dimension()
    Vbar_ele_space_dim = Vbar.dolfin_element().space_dimension()
    num_facets = mesh.ufl_cell().num_facets()

    xbar = Function(Vbar)
    xbar.vector.set(1)
    c = dolfinx_hdg.assemble.pack_facet_space_coeffs_cellwise(xbar, mesh)

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
        xbar = numba.carray(w_, (num_facets * Vbar_ele_space_dim),
                            dtype=PETSc.ScalarType)

        for i in range(V_ele_space_dim):
            for j in range(Vbar_ele_space_dim):
                b[i] += 1 / Vbar_ele_space_dim * xbar[Vbar_ele_space_dim * i + j]

    integrals = {dolfinx.fem.IntegralType.cell:
                 ([(-1, tabulate_tensor.address)], None)}
    f = dolfinx.cpp.fem.Form(
        [V._cpp_object], integrals, [], [], False, None)

    b = dolfinx.fem.assemble_vector(f, coeffs=(None, c))
    b.assemble()

    b_expected = np.array([1, 1, 1, 1, 1, 1])

    assert(np.allclose(b[:], b_expected))
