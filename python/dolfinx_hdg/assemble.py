import functools
from petsc4py import PETSc
import dolfinx
import typing
from dolfinx.fem.dirichletbc import DirichletBC
from dolfinx.fem.form import Form
from dolfinx.fem.assemble import (_create_cpp_form, _cpp_dirichletbc,
                                  pack_constants, Coefficients)
import dolfinx_hdg.cpp
import numpy as np


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
         coeffs[1] if coeffs[1] is not None else dolfinx_hdg.cpp.pack_coefficients(_L))
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
         coeffs[1] if coeffs[1] is not None else dolfinx_hdg.cpp.pack_coefficients(_L))
    with b.localForm() as b_local:
        dolfinx_hdg.cpp.assemble_vector(b_local.array_w, _L, c[0], c[1])
    return b


# NOTE This assumes the facet space is in the a[1][1] position
@functools.singledispatch
def assemble_matrix(a: Form,
        bcs: typing.List[DirichletBC] = [],
        diagonal: float = 1.0,
        coeffs=Coefficients(None, None)) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    # Need to call dolfinx_hdg.create_matrix as the condensed system
    # has a different sparsity pattern to what dolfinx.create_matrix
    # would provide
    A = dolfinx_hdg.cpp.create_matrix(_create_cpp_form(a))
    return assemble_matrix(A, a, bcs, diagonal, coeffs)


@assemble_matrix.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: Form,
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0,
      coeffs=Coefficients(None, None)) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    _a = _create_cpp_form(a)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(_a),
         coeffs[1] if coeffs[1] is not None else dolfinx_hdg.cpp.pack_coefficients(_a))
    dolfinx_hdg.cpp.assemble_matrix_petsc(A, _a, c[0], c[1], _cpp_dirichletbc(bcs))
    # TODO When will this not be true? Mixed facet HDG terms?
    if _a.function_spaces[0].id == _a.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.insert_diagonal(A, _a.function_spaces[0],
                                        _cpp_dirichletbc(bcs), diagonal)
    return A
