import functools
from petsc4py import PETSc
import dolfinx
import dolfinx.cpp
import dolfinx_hdg.cpp
from dolfinx.fem.forms import FormMetaClass
import collections
import contextlib
from dolfinx.fem.assemble import pack_constants


def apply_lifting(b, a, bcs, x0=[], scale=1.0, constants=None, coeffs=None):
    """Apply the function :func:`dolfinx.fem.apply_lifting` to a PETSc Vector."""
    with contextlib.ExitStack() as stack:
        x0 = [stack.enter_context(x.localForm()) for x in x0]
        x0_r = [x.array_r for x in x0]
        b_local = stack.enter_context(b.localForm())
        apply_lifting_array(b_local.array_w, a, bcs, x0_r,
                            scale, constants, coeffs)


def apply_lifting_array(b, a, bcs, x0=None, scale=1.0,
                        constants=None, coeffs=None):
    """Modify RHS vector b for lifting of Dirichlet boundary conditions.

    It modifies b such that:

    .. math::

        b \\leftarrow  b - \\text{scale} * A_j (g_j - x0_j)

    where j is a block (nest) index. For a non-blocked problem j = 0.
    The boundary conditions bcs are on the trial spaces V_j. The forms
    in [a] must have the same test space as L (from which b was built),
    but the trial space may differ. If x0 is not supplied, then it is
    treated as zero.

    Note:
        Ghost contributions are not accumulated (not sent to owner).
        Caller is responsible for calling VecGhostUpdateBegin/End.

    """
    x0 = [] if x0 is None else x0
    constants = [form and pack_constants(
        form) for form in a] if constants is None else constants
    coeffs = [{} if form is None else pack_coefficients(
        form) for form in a] if coeffs is None else coeffs

    dolfinx_hdg.cpp.apply_lifting(b, a, constants, coeffs, bcs, x0, scale)


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
            return dolfinx_hdg.cpp.pack_coefficients(form)

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
        constants = pack_constants(L) if constants is None else constants
        coeffs = pack_coefficients(L) if coeffs is None else coeffs
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
    constants = pack_constants(a) if constants is None else constants
    coeffs = pack_coefficients(a) if coeffs is None else coeffs

    dolfinx_hdg.cpp.assemble_matrix(A, a, constants, coeffs, bcs)
    if a.function_spaces[0] is a.function_spaces[1]:
        A.assemblyBegin(PETSc.Mat.AssemblyType.FLUSH)
        A.assemblyEnd(PETSc.Mat.AssemblyType.FLUSH)
        dolfinx.cpp.fem.petsc.insert_diagonal(
            A, a.function_spaces[0], bcs, diagonal)
    return A


@functools.singledispatch
def assemble_matrix_block(a,
                          bcs=[],
                          diagonal=1.0,
                          constants=None, coeffs=None):
    return _assemble_matrix_block_form(a, bcs, diagonal, constants, coeffs)


assemble_matrix_block.register(list)


def _assemble_matrix_block_form(a,
                                bcs=[],
                                diagonal=1.0,
                                constants=None, coeffs=None):
    """Assemble bilinear forms into matrix"""
    A = dolfinx_hdg.cpp.create_matrix_block(a)
    return _assemble_matrix_block_mat(A, a, bcs, diagonal, constants, coeffs)


@assemble_matrix_block.register
def _assemble_matrix_block_mat(A: PETSc.Mat, a,
                               bcs=[], diagonal: float = 1.0,
                               constants=None, coeffs=None):
    """Assemble bilinear forms into matrix"""
    constants = [[form and pack_constants(form) for form in forms]
                 for forms in a] if constants is None else constants
    coeffs = [[{} if form is None else pack_coefficients(
        form) for form in forms] for forms in a] if coeffs is None else coeffs

    V = dolfinx.fem.petsc._extract_function_spaces(a)
    is_rows = dolfinx.cpp.la.petsc.create_index_sets(
        [(Vsub.dofmap.index_map, Vsub.dofmap.index_map_bs) for Vsub in V[0]])
    is_cols = dolfinx.cpp.la.petsc.create_index_sets(
        [(Vsub.dofmap.index_map, Vsub.dofmap.index_map_bs) for Vsub in V[1]])

    # Assemble form
    for i, a_row in enumerate(a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                dolfinx_hdg.cpp.assemble_matrix(
                    Asub, a_sub, constants[i][j], coeffs[i][j], bcs, True)
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)
            elif i == j:
                for bc in bcs:
                    row_forms = [
                        row_form for row_form in a_row if row_form is not None]
                    assert len(row_forms) > 0
                    if row_forms[0].function_spaces[0].contains(bc.function_space):
                        raise RuntimeError(
                            f"Diagonal sub-block ({i}, {j}) cannot be 'None' and have DirichletBC applied."
                            " Consider assembling a zero block.")

    # Flush to enable switch from add to set in the matrix
    A.assemble(PETSc.Mat.AssemblyType.FLUSH)

    # Set diagonal
    for i, a_row in enumerate(a):
        for j, a_sub in enumerate(a_row):
            if a_sub is not None:
                Asub = A.getLocalSubMatrix(is_rows[i], is_cols[j])
                if a_sub.function_spaces[0] is a_sub.function_spaces[1]:
                    dolfinx.cpp.fem.petsc.insert_diagonal(
                        Asub, a_sub.function_spaces[0], bcs, diagonal)
                A.restoreLocalSubMatrix(is_rows[i], is_cols[j], Asub)

    return A


@functools.singledispatch
def assemble_vector_block(L, a, bcs=[], x0=None, scale=1.0,
                          constants_L=None, coeffs_L=None,
                          constants_a=None, coeffs_a=None):
    return _assemble_vector_block_form(L, a, bcs, x0, scale, constants_L, coeffs_L, constants_a, coeffs_a)


@assemble_vector_block.register(list)
def _assemble_vector_block_form(L, a, bcs=[], x0=None, scale=1.0,
                                constants_L=None, coeffs_L=None,
                                constants_a=None, coeffs_a=None):
    """Assemble linear forms into a monolithic vector. The vector is not
    finalised, i.e. ghost values are not accumulated.

    """
    maps = [(form.function_spaces[0].dofmap.index_map,
             form.function_spaces[0].dofmap.index_map_bs) for form in L]
    b = dolfinx.cpp.fem.petsc.create_vector_block(maps)
    with b.localForm() as b_local:
        b_local.set(0.0)
    return _assemble_vector_block_vec(b, L, a, bcs, x0, scale, constants_L, coeffs_L,
                                      constants_a, coeffs_a)


@assemble_vector_block.register
def _assemble_vector_block_vec(b: PETSc.Vec, L, a, bcs=[], x0=None,
                               scale=1.0,
                               constants_L=None, coeffs_L=None,
                               constants_a=None, coeffs_a=None):
    """Assemble linear forms into a monolithic vector. The vector is not
    zeroed and it is not finalised, i.e. ghost values are not
    accumulated.

    """
    maps = [(form.function_spaces[0].dofmap.index_map,
             form.function_spaces[0].dofmap.index_map_bs) for form in L]
    if x0 is not None:
        x0_local = dolfinx.cpp.la.petsc.get_local_vectors(x0, maps)
        x0_sub = x0_local
    else:
        x0_local = []
        x0_sub = [None] * len(maps)

    constants_L = [form and pack_constants(
        form) for form in L] if constants_L is None else constants_L
    coeffs_L = [{} if form is None else pack_coefficients(
        form) for form in L] if coeffs_L is None else coeffs_L
    constants_a = [[form and pack_constants(form) for form in forms]
                   for forms in a] if constants_a is None else constants_a
    coeffs_a = [[{} if form is None else pack_coefficients(
        form) for form in forms] for forms in a] if coeffs_a is None else coeffs_a

    bcs1 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.forms.extract_function_spaces(a, 1), bcs)
    b_local = dolfinx.cpp.la.petsc.get_local_vectors(b, maps)
    for b_sub, L_sub, a_sub, const_L, coeff_L, const_a, coeff_a in zip(b_local, L, a,
                                                                       constants_L, coeffs_L,
                                                                       constants_a, coeffs_a):
        dolfinx_hdg.cpp.assemble_vector(b_sub, L_sub, const_L, coeff_L)
        dolfinx_hdg.cpp.apply_lifting(b_sub, a_sub, const_a,
                                      coeff_a, bcs1, x0_local, scale)

    dolfinx.cpp.la.petsc.scatter_local_vectors(b, b_local, maps)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bcs0 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.forms.extract_function_spaces(L), bcs)
    offset = 0
    b_array = b.getArray(readonly=False)
    for submap, bc, _x0 in zip(maps, bcs0, x0_sub):
        size = submap[0].size_local * submap[1]
        dolfinx.cpp.fem.set_bc(b_array[offset: offset + size], bc, _x0, scale)
        offset += size

    return b
