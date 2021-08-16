import functools
from petsc4py import PETSc
import dolfinx
import typing
from dolfinx.fem.dirichletbc import DirichletBC
from dolfinx.fem.form import Form
from dolfinx.fem.assemble import _create_cpp_form
from dolfinx_hdg.cpp import create_sparsity_pattern

@functools.singledispatch
def assemble_matrix(a: typing.Union[Form, dolfinx.cpp.fem.Form],
                    bcs: typing.List[DirichletBC] = [],
                    diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    cpp_form = _create_cpp_form(a)
    sp = create_sparsity_pattern(cpp_form)
    A = dolfinx.cpp.la.create_matrix(cpp_form.mesh.mpi_comm(), sp)
    return assemble_matrix(A, a, bcs, diagonal)


@assemble_matrix.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a: typing.Union[Form, dolfinx.cpp.fem.Form],
      bcs: typing.List[DirichletBC] = [],
      diagonal: float = 1.0) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    _a = _create_cpp_form(a)
    dolfinx.cpp.fem.assemble_matrix_petsc(A, _a, bcs)
    if _a.function_spaces[0].id == _a.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.insert_diagonal(A, _a.function_spaces[0], bcs, diagonal)
    return A