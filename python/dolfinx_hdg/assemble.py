import functools
from petsc4py import PETSc
import dolfinx
import typing
from dolfinx.fem.dirichletbc import DirichletBC
from dolfinx.fem.form import Form
from dolfinx.fem.assemble import _create_cpp_form
import dolfinx_hdg.cpp


# NOTE This assumes the facet space is in the L[1] position
# TODO DOLFINx also has an assemble_vector for assembling into an
# existing PETSc vector, using @functools.singledispatch. Implement
# this!
def assemble_vector(L: typing.List[
        typing.Union[Form, dolfinx.cpp.fem.Form]]) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector. The returned vector is
    not finalised, i.e. ghost values are not accumulated on the owning
    processes.

    """
    assert(len(L) == 2)
    _L = _create_cpp_form(L)
    b = dolfinx.cpp.la.create_vector(
        _L[1].function_spaces[0].dofmap.index_map,
        _L[1].function_spaces[0].dofmap.index_map_bs)
    with b.localForm() as b_local:
        b_local.set(0.0)
        # FIXME Pass L
        dolfinx_hdg.cpp.assemble_vector(b_local.array_w, _L[1])
    return b


# NOTE This assumes the facet space is in the a[1][1] position
@functools.singledispatch
def assemble_matrix(a: typing.List[typing.List[
        typing.Union[Form, dolfinx.cpp.fem.Form]]],
        bcs: typing.List[DirichletBC] = [],
        diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    # FIXME Check a is 2x2
    # Need to call dolfinx_hdg.create_matrix as the condensed system
    # has a different sparsity pattern to what dolfinx.create_matrix
    # would provide
    A = dolfinx_hdg.cpp.create_matrix(_create_cpp_form(a[1][1]))
    return assemble_matrix(A, a, bcs, diagonal)


@assemble_matrix.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.List[typing.List[
                     typing.Union[Form, dolfinx.cpp.fem.Form]]],
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    _a = _create_cpp_form(a)
    dolfinx_hdg.cpp.assemble_matrix_petsc(A, _a, bcs)
    # TODO When will this not be true? Mixed facet HDG terms?
    if _a[1][1].function_spaces[0].id == _a[1][1].function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.insert_diagonal(A, _a[1][1].function_spaces[0],
                                        bcs, diagonal)
    return A
