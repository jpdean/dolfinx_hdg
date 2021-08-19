import typing
from petsc4py import PETSc
from dolfinx.fem.form import Form
import dolfinx
from dolfinx.fem.assemble import _create_cpp_form
import dolfinx_hdg.cpp


def back_sub(xbar: PETSc.Vec,
             a: typing.List[typing.List[
                 typing.Union[Form, dolfinx.cpp.fem.Form]]],
             L: typing.List[
                 typing.Union[Form, dolfinx.cpp.fem.Form]]) -> PETSc.Vec:
    # TODO Also check size of a
    assert(len(L) == 2)
    _L = _create_cpp_form(L)
    _a = _create_cpp_form(a)
    x = dolfinx.cpp.la.create_vector(
        _L[0].function_spaces[0].dofmap.index_map,
        _L[0].function_spaces[0].dofmap.index_map_bs)
    # TODO Is this correct?
    with x.localForm() as x_local, xbar.localForm() as xbar_local:
        x_local.set(0.0)
        dolfinx_hdg.cpp.back_sub(x_local.array_w, xbar_local.array_w, _a, _L)
    return x
