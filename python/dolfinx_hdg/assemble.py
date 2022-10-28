import functools
from petsc4py import PETSc
import dolfinx
import dolfinx.cpp
import dolfinx_hdg.cpp
from dolfinx.fem.forms import FormMetaClass
import collections
from dolfinx_hdg.cpp import pack_coefficients as _pack_coefficients


# @functools.singledispatch
# def assemble_vector(L, mesh, facet_mesh,
#                     coeffs=Coefficients(None, None)) -> PETSc.Vec:
#     """Assemble linear form into a new PETSc vector. The returned vector is
#     not finalised, i.e. ghost values are not accumulated on the owning
#     processes.

#     """
#     b = dolfinx.la.create_petsc_vector(
#         L.function_spaces[0].dofmap.index_map,
#         L.function_spaces[0].dofmap.index_map_bs)
#     c = (coeffs[0] if coeffs[0] is not None else pack_constants(L),
#          coeffs[1] if coeffs[1] is not None else dolfinx_hdg.cpp.pack_coefficients(L))
#     with b.localForm() as b_local:
#         b_local.set(0.0)
#         dolfinx_hdg.cpp.assemble_vector(
#             b_local.array_w, L, mesh, facet_mesh, c[0], c[1])
#     return b


# @assemble_vector.register(PETSc.Vec)
# def _(b: PETSc.Vec, L, mesh, facet_mesh,
#       coeffs=Coefficients(None, None)) -> PETSc.Vec:
#     """Assemble linear form into an existing PETSc vector. The vector is not
#     zeroed before assembly and it is not finalised, qi.e. ghost values are
#     not accumulated on the owning processes.
#     """
#     c = (coeffs[0] if coeffs[0] is not None else pack_constants(L),
#          coeffs[1] if coeffs[1] is not None else dolfinx_hdg.cpp.pack_coefficients(L))
#     with b.localForm() as b_local:
#         dolfinx_hdg.cpp.assemble_vector(
#             b_local.array_w, L, mesh, facet_mesh, c[0], c[1])
#     return b


def pack_coefficients(form):
    """Compute form coefficients.

    Pack the `coefficients` that appear in forms. The packed
    coefficients can be passed to an assembler. This is a
    performance optimisation for cases where a form is assembled
    multiple times and (some) coefficients do not change.

    If ``form`` is an array of forms, this function returns an array of
    form coefficients with the same shape as form.

    Args:
        form: A single form or array of forms to pack the constants for.

    Returns:
        Coefficients for each form.

    """
    def _pack(form):
        if form is None:
            return {}
        elif isinstance(form, collections.abc.Iterable):
            return list(map(lambda sub_form: _pack(sub_form), form))
        else:
            return _pack_coefficients(form)

    return _pack(form)


@functools.singledispatch
def assemble_vector(L, constants=None, coeffs=None):
    return _assemble_vector_form(L, constants, coeffs)


@assemble_vector.register(FormMetaClass)
def _assemble_vector_form(L, constants=None, coeffs=None):
    """Assemble linear form into a new PETSc vector.

    Note:
        The returned vector is not finalised, i.e. ghost values are not
        accumulated on the owning processes.

    Args:
        L: A linear form.

    Returns:
        An assembled vector.

    """
    b = dolfinx.la.create_petsc_vector(L.function_spaces[0].dofmap.index_map,
                                       L.function_spaces[0].dofmap.index_map_bs)
    with b.localForm() as b_local:
        # TODO Pack consts and coeffs!
        # constants = _pack_constants(L) if constants is None else constants
        # coeffs = _pack_coefficients(L) if coeffs is None else coeffs
        constants = []
        import numpy as np
        coeffs = {(dolfinx.fem.IntegralType.cell, -1):
                  np.zeros(shape=(0, 0), dtype=np.float64)}
        print("TODO Assemble vec")
        dolfinx_hdg.cpp.assemble_vector(b_local.array_w, L, constants, coeffs)
    return b


@functools.singledispatch
def assemble_matrix(a, bcs=[], diagonal=1.0,
                    constants=None, coeffs=None):
    return _assemble_matrix_form(a, bcs, diagonal, constants, coeffs)


@assemble_matrix.register(FormMetaClass)
def _assemble_matrix_form(a, bcs=[],
                          diagonal=1.0,
                          constants=None, coeffs=None):
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.
    """
    A = dolfinx_hdg.cpp.create_matrix(a)
    _assemble_matrix_mat(A, a, bcs, diagonal, constants, coeffs)
    return A


@assemble_matrix.register
def _assemble_matrix_mat(A: PETSc.Mat, a, bcs=[],
                         diagonal=1.0, constants=None, coeffs=None):
    """Assemble bilinear form into a matrix. The returned matrix is not
    finalised, i.e. ghost values are not accumulated.

    """

    # TODO Pack constants and coeffs
    # constants = pack_constants(a) if constants is None else constants
    # coeffs = pack_coefficients(a) if coeffs is None else coeffs

    constants = []
    import numpy as np
    coeffs = {(dolfinx.fem.IntegralType.cell, -1)
               : np.zeros(shape=(0, 0), dtype=np.float64)}

    dolfinx_hdg.cpp.assemble_matrix(A, a, constants, coeffs, bcs)
    if a.function_spaces[0] is a.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.petsc.insert_diagonal(
            A, a.function_spaces[0], bcs, diagonal)
    return A

# @assemble_matrix.register(PETSc.Mat)
# def _(A: PETSc.Mat,
#       a,
#       mesh, facet_mesh,
#       bcs = [],
#       diagonal: float = 1.0,
#       coeffs=Coefficients(None, None)) -> PETSc.Mat:
#     """Assemble bilinear form into a matrix. The returned matrix is not
#     finalised, i.e. ghost values are not accumulated.

#     """
#     c = (coeffs[0] if coeffs[0] is not None else pack_constants(a),
#          coeffs[1] if coeffs[1] is not None else dolfinx_hdg.cpp.pack_coefficients(a))
#     dolfinx_hdg.cpp.assemble_matrix_petsc(A, a, mesh, facet_mesh, c[0], c[1], bcs)
#     # TODO When will this not be true? Mixed facet HDG terms?
#     if a.function_spaces[0].id == a.function_spaces[1].id:
#         A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
#         A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
#         dolfinx.cpp.fem.petsc.insert_diagonal(A, a.function_spaces[0],
#                                               bcs, diagonal)
#     return A
