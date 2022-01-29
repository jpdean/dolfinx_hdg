import functools
from petsc4py import PETSc
import dolfinx
from dolfinx.fem.assemble import (pack_constants, Coefficients)
import dolfinx.cpp
import dolfinx_hdg.cpp


@functools.singledispatch
def assemble_vector(L, mesh, facet_mesh,
                    coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear form into a new PETSc vector. The returned vector is
    not finalised, i.e. ghost values are not accumulated on the owning
    processes.

    """
    b = dolfinx.la.create_petsc_vector(
        L.function_spaces[0].dofmap.index_map,
        L.function_spaces[0].dofmap.index_map_bs)
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(L),
         coeffs[1] if coeffs[1] is not None else dolfinx_hdg.cpp.pack_coefficients(L))
    with b.localForm() as b_local:
        b_local.set(0.0)
        dolfinx_hdg.cpp.assemble_vector(
            b_local.array_w, L, mesh, facet_mesh, c[0], c[1])
    return b


@assemble_vector.register(PETSc.Vec)
def _(b: PETSc.Vec, L, mesh, facet_mesh,
      coeffs=Coefficients(None, None)) -> PETSc.Vec:
    """Assemble linear form into an existing PETSc vector. The vector is not
    zeroed before assembly and it is not finalised, qi.e. ghost values are
    not accumulated on the owning processes.
    """
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(L),
         coeffs[1] if coeffs[1] is not None else dolfinx_hdg.cpp.pack_coefficients(L))
    with b.localForm() as b_local:
        dolfinx_hdg.cpp.assemble_vector(
            b_local.array_w, mesh, facet_mesh, L, c[0], c[1])
    return b


# NOTE This assumes the facet space is in the a[1][1] position
@functools.singledispatch
def assemble_matrix(a,
        mesh, facet_mesh,
        bcs = [],
        diagonal: float = 1.0,
        coeffs=Coefficients(None, None)) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    # Need to call dolfinx_hdg.create_matrix as the condensed system
    # has a different sparsity pattern to what dolfinx.create_matrix
    # would provide
    A = dolfinx_hdg.cpp.create_matrix(a)
    return assemble_matrix(A, a, mesh, facet_mesh, bcs, diagonal, coeffs)


@assemble_matrix.register(PETSc.Mat)
def _(A: PETSc.Mat,
      a,
      mesh, facet_mesh,
      bcs = [],
      diagonal: float = 1.0,
      coeffs=Coefficients(None, None)) -> PETSc.Mat:
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """
    c = (coeffs[0] if coeffs[0] is not None else pack_constants(a),
         coeffs[1] if coeffs[1] is not None else dolfinx_hdg.cpp.pack_coefficients(a))
    dolfinx_hdg.cpp.assemble_matrix_petsc(A, a, mesh, facet_mesh, c[0], c[1], bcs)
    # TODO When will this not be true? Mixed facet HDG terms?
    if a.function_spaces[0].id == a.function_spaces[1].id:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.petsc.insert_diagonal(A, a.function_spaces[0],
                                              bcs, diagonal)
    return A
