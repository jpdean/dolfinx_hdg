import functools
from petsc4py import PETSc
import dolfinx
import typing
from dolfinx.fem.dirichletbc import DirichletBC
from dolfinx.fem.form import Form
from dolfinx.fem.assemble import (_create_cpp_form, _cpp_dirichletbc,
                                  pack_constants, pack_coefficients,
                                  Coefficients)
import dolfinx_hdg.cpp
import numpy as np


# FIXME This is a HACK. Either make C++ implementation or
# find a better approach
def pack_facet_space_coeffs_cellwise(coeff, mesh):
    print("WARNING: pack_facet_space_coeffs_cellwise will be replaced")
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, tdim - 1)
    c_to_f = mesh.topology.connectivity(tdim, tdim - 1)
    c_to_v = mesh.topology.connectivity(2, 0)
    f_to_v = mesh.topology.connectivity(1, 0)
    num_cells = c_to_f.num_nodes
    num_cell_facets = mesh.ufl_cell().num_facets()

    V = coeff.function_space
    dofmap = V.dofmap.list
    ele_space_dim = V.dolfin_element().space_dimension()
    packed_coeffs = np.zeros((num_cells, num_cell_facets * ele_space_dim))
    for cell in range(num_cells):
        cell_facets = c_to_f.links(cell)
        cell_vertices = c_to_v.links(cell)
        for local_facet in range(num_cell_facets):
            facet = cell_facets[local_facet]
            facet_vertices = f_to_v.links(cell_facets[local_facet])
            flip_dofs = \
                not np.allclose(cell_vertices[np.arange(len(cell_vertices))!=local_facet], facet_vertices)
            dofs = dofmap.links(facet)
            if flip_dofs:
                dofs = np.flip(dofs)
            for i in range(ele_space_dim):
                packed_coeffs[cell, local_facet * ele_space_dim + i] = \
                    coeff.vector[dofs[i]]

    return packed_coeffs


@functools.singledispatch
def assemble_vector(L: Form, coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector. The returned vector is
    not finalised, i.e. ghost values are not accumulated on the owning
    processes.

    """
    _L = _create_cpp_form(L)
    b = dolfinx.cpp.la.create_vector(
        _L.function_spaces[0].dofmap.index_map,
        _L.function_spaces[0].dofmap.index_map_bs)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_L),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_L))
    with b.localForm() as b_local:
        b_local.set(0.0)
        dolfinx_hdg.cpp.assemble_vector(b_local.array_w, _L, c[0], c[1])
    return b


@assemble_vector.register(PETSc.Vec)
def _(b: PETSc.Vec, L: Form, coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear form into an existing PETSc vector. The vector is not
    zeroed before assembly and it is not finalised, qi.e. ghost values are
    not accumulated on the owning processes.
    """
    _L = _create_cpp_form(L)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_L),
         coeffs[1] if coeffs[1] is not None else pack_coefficients(_L))
    with b.localForm() as b_local:
        dolfinx_hdg.cpp.assemble_vector(b_local.array_w, _L, c[0], c[1])
    return b


# NOTE This assumes the facet space is in the a[1][1] position
@functools.singledispatch
def assemble_matrix(a: Form,
        bcs: typing.List[DirichletBC] = [],
        diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    # Need to call dolfinx_hdg.create_matrix as the condensed system
    # has a different sparsity pattern to what dolfinx.create_matrix
    # would provide
    A = dolfinx_hdg.cpp.create_matrix(_create_cpp_form(a))
    return assemble_matrix(A, a, bcs, diagonal)


@assemble_matrix.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: Form,
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    _a = _create_cpp_form(a)
    dolfinx_hdg.cpp.assemble_matrix_petsc(A, _a, _cpp_dirichletbc(bcs))
    # TODO When will this not be true? Mixed facet HDG terms?
    if _a.function_spaces[0].id == _a.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.insert_diagonal(A, _a.function_spaces[0],
                                        _cpp_dirichletbc(bcs), diagonal)
    return A
