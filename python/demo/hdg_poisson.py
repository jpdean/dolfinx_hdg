# TODO Add fastmath flag to numba (see custom assembler demos)

from dolfinx.cpp.fem.petsc import insert_diagonal
from dolfinx.cpp.la.petsc import create_matrix
from dolfinx.cpp.la import SparsityPattern
from mpi4py import MPI
from dolfinx import mesh, fem, jit, io
from dolfinx.cpp.mesh import cell_num_entities
import numpy as np
import ufl
from ufl import inner, grad, dot, div
from petsc4py import PETSc
from dolfinx.fem import IntegralType
import cffi
import numba
from dolfinx.cpp.fem import Form_float64
import sys
from dolfinx.common import Timer, list_timings, TimingType
from dolfinx_hdg.assemble import assemble_matrix as assemble_matrix_hdg


def main():
    def norm_L2(comm, v):
        return np.sqrt(comm.allreduce(fem.assemble_scalar(
            fem.form(ufl.inner(v, v) * ufl.dx)), op=MPI.SUM))

    def reorder_mesh(msh):
        # FIXME Check this is correct
        # FIXME For a high-order mesh, the geom has more dofs so need to modify
        # this
        # FIXME What about quads / hexes?
        tdim = msh.topology.dim
        num_cell_vertices = cell_num_entities(msh.topology.cell_type, 0)
        c_to_v = msh.topology.connectivity(tdim, 0)
        geom_dofmap = msh.geometry.dofmap
        vertex_imap = msh.topology.index_map(0)
        geom_imap = msh.geometry.index_map()
        for i in range(0, len(c_to_v.array), num_cell_vertices):
            topo_perm = np.argsort(vertex_imap.local_to_global(
                c_to_v.array[i:i+num_cell_vertices]))
            geom_perm = np.argsort(geom_imap.local_to_global(
                geom_dofmap.array[i:i+num_cell_vertices]))

            c_to_v.array[i:i+num_cell_vertices] = \
                c_to_v.array[i:i+num_cell_vertices][topo_perm]
            geom_dofmap.array[i:i+num_cell_vertices] = \
                geom_dofmap.array[i:i+num_cell_vertices][geom_perm]

    def boundary(x):
        lr = np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
        tb = np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0)
        lrtb = lr | tb
        if tdim == 2:
            return lrtb
        else:
            assert tdim == 3
            fb = np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0)
            return lrtb | fb

    def par_print(string):
        if comm.rank == 0:
            print(string)
            sys.stdout.flush()

    comm = MPI.COMM_WORLD

    par_print("Create mesh")
    with Timer("Create mesh") as t:
        n = 2
        # n = round((500000 * comm.size / 60)**(1 / 3))
        par_print(f"n = {n}")
        msh = mesh.create_unit_square(
            comm, n, n, ghost_mode=mesh.GhostMode.none)
        # msh = mesh.create_unit_cube(
        #     comm, n, n, n, ghost_mode=mesh.GhostMode.none)
        reorder_mesh(msh)

    par_print("Create submesh")
    with Timer("Create submesh") as t:
        tdim = msh.topology.dim
        fdim = tdim - 1
        num_cell_facets = cell_num_entities(msh.topology.cell_type, fdim)
        msh.topology.create_entities(fdim)
        facet_imap = msh.topology.index_map(fdim)
        num_facets = facet_imap.size_local + facet_imap.num_ghosts
        facets = np.arange(num_facets, dtype=np.int32)

        # NOTE Despite all facets being present in the submesh, the entity map isn't
        # necessarily the identity in parallel
        facet_mesh, entity_map = mesh.create_submesh(msh, fdim, facets)[0:2]

    par_print("Create function spaces")
    with Timer("Create function spaces") as t:
        k = 1
        V = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
        Vbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

    V_ele_space_dim = V.element.space_dimension
    Vbar_ele_space_dim = Vbar.element.space_dimension

    num_cell_facets = msh.ufl_cell().num_facets()
    num_dofs_g = len(msh.geometry.dofmap.links(0))

    par_print("Define problem")
    with Timer("Define problem") as t:
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        ubar = ufl.TrialFunction(Vbar)
        vbar = ufl.TestFunction(Vbar)

        h = ufl.CellDiameter(msh)
        n = ufl.FacetNormal(msh)
        gamma = 16.0 * k**2 / h

        dx_c = ufl.Measure("dx", domain=msh)
        ds_c = ufl.Measure("ds", domain=msh)

        a_00 = inner(grad(u), grad(v)) * dx_c - \
            (inner(u, dot(grad(v), n)) * ds_c +
             inner(v, dot(grad(u), n)) * ds_c) + \
            gamma * inner(u, v) * ds_c
        a_10 = inner(dot(grad(u), n) - gamma * u, vbar) * ds_c
        a_11 = gamma * inner(ubar, vbar) * ds_c

        x = ufl.SpatialCoordinate(msh)
        u_e = 1
        for i in range(tdim):
            u_e *= ufl.sin(ufl.pi * x[i])
        f = - div(grad(u_e))
        L_0 = inner(f, v) * dx_c

    par_print("Create dofmap, inv ent map, and connectivities")
    with Timer("Create dofmap, inv ent map, and connectivities") as t:
        num_owned_cells = msh.topology.index_map(msh.topology.dim).size_local
        num_cells = num_owned_cells + \
            msh.topology.index_map(msh.topology.dim).num_ghosts
        x_dofs = msh.geometry.dofmap.array.reshape(num_cells, num_dofs_g)
        x = msh.geometry.x
        Vbar_dofmap = Vbar.dofmap.list.array.reshape(num_facets, Vbar_ele_space_dim).astype(
            np.dtype(PETSc.IntType))

        inv_entity_map = np.full_like(entity_map, -1)
        for i, f in enumerate(entity_map):
            inv_entity_map[f] = i

        c_to_f = msh.topology.connectivity(
            msh.topology.dim, msh.topology.dim - 1)
        c_to_facet_mesh_f = inv_entity_map[c_to_f.array].reshape(
            num_cells, num_cell_facets)

    nptype = "float64"
    ffcxtype = "double"

    par_print("JIT kernels")
    with Timer("JIT kernels") as t:
        # LHS forms
        ufcx_form_00, _, _ = jit.ffcx_jit(
            msh.comm, a_00, form_compiler_options={"scalar_type": ffcxtype})
        kernel_00_cell = getattr(ufcx_form_00.integrals(IntegralType.cell)[0],
                                 f"tabulate_tensor_{nptype}")
        kernel_00_facet = getattr(ufcx_form_00.integrals(IntegralType.exterior_facet)[0],
                                  f"tabulate_tensor_{nptype}")
        ufcx_form_10, _, _ = jit.ffcx_jit(
            msh.comm, a_10, form_compiler_options={"scalar_type": ffcxtype})
        kernel_10 = getattr(ufcx_form_10.integrals(IntegralType.exterior_facet)[0],
                            f"tabulate_tensor_{nptype}")
        ufcx_form_11, _, _ = jit.ffcx_jit(
            msh.comm, a_11, form_compiler_options={"scalar_type": ffcxtype})
        kernel_11 = getattr(ufcx_form_11.integrals(IntegralType.exterior_facet)[0],
                            f"tabulate_tensor_{nptype}")

        # RHS forms
        ufcx_form_0, _, _ = jit.ffcx_jit(
            msh.comm, L_0, form_compiler_options={"scalar_type": ffcxtype})
        kernel_0 = getattr(ufcx_form_0.integrals(IntegralType.cell)[0],
                           f"tabulate_tensor_{nptype}")

    par_print("FFI setup")
    with Timer("FFI setup") as t:
        # Get PETSc int and scalar types
        if np.dtype(PETSc.ScalarType).kind == 'c':
            complex = True
        else:
            complex = False

        scalar_size = np.dtype(PETSc.ScalarType).itemsize
        index_size = np.dtype(PETSc.IntType).itemsize

        if index_size == 8:
            c_int_t = "int64_t"
        elif index_size == 4:
            c_int_t = "int32_t"
        else:
            raise RuntimeError(
                f"Cannot translate PETSc index size into a C type, index_size: {index_size}.")

        if complex and scalar_size == 16:
            c_scalar_t = "double _Complex"
        elif complex and scalar_size == 8:
            c_scalar_t = "float _Complex"
        elif not complex and scalar_size == 8:
            c_scalar_t = "double"
        elif not complex and scalar_size == 4:
            c_scalar_t = "float"
        else:
            raise RuntimeError(
                f"Cannot translate PETSc scalar type to a C type, complex: {complex} size: {scalar_size}.")

        ffi = cffi.FFI()
        # Get MatSetValuesLocal from PETSc available via cffi in ABI mode
        ffi.cdef("""int MatSetValuesLocal(void* mat, {0} nrow, const {0}* irow,
                                        {0} ncol, const {0}* icol, const {1}* y, int addv);
        """.format(c_int_t, c_scalar_t))
        petsc_lib_cffi = fem.petsc.load_petsc_lib(ffi.dlopen)
        MatSetValues_abi = petsc_lib_cffi.MatSetValuesLocal

        # FIXME See if there is a better way to pass null
        null64 = np.zeros(0, dtype=np.float64)
        null32 = np.zeros(0, dtype=np.int32)
        null8 = np.zeros(0, dtype=np.uint8)

    @numba.njit(fastmath=True)
    def compute_A11(coords):
        A11 = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                        num_cell_facets * Vbar_ele_space_dim),
                       dtype=PETSc.ScalarType)
        A11_f = np.zeros((Vbar_ele_space_dim, Vbar_ele_space_dim),
                         dtype=PETSc.ScalarType)

        entity_local_index = np.zeros((1), dtype=np.int32)
        for local_f in range(num_cell_facets):
            entity_local_index[0] = local_f
            A11_f.fill(0.0)

            kernel_11(ffi.from_buffer(A11_f),
                      ffi.from_buffer(null64),
                      ffi.from_buffer(null64),
                      ffi.from_buffer(coords),
                      ffi.from_buffer(entity_local_index),
                      ffi.from_buffer(null8))
            # Insert in correct location
            start = local_f * Vbar_ele_space_dim
            end = start + Vbar_ele_space_dim
            A11[start:end, start:end] += A11_f[:, :]
        return A11

    @numba.njit(fastmath=True)
    def compute_A00_A10(coords):
        A00 = np.zeros((V_ele_space_dim, V_ele_space_dim),
                       dtype=PETSc.ScalarType)
        A10 = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                        V_ele_space_dim), dtype=PETSc.ScalarType)
        A10_f = np.zeros((Vbar_ele_space_dim, V_ele_space_dim),
                         dtype=PETSc.ScalarType)

        kernel_00_cell(ffi.from_buffer(A00),
                       ffi.from_buffer(null64),
                       ffi.from_buffer(null64),
                       ffi.from_buffer(coords),
                       ffi.from_buffer(null32),
                       ffi.from_buffer(null8))
        # FIXME Is there a neater way to do this?
        entity_local_index = np.zeros((1), dtype=np.int32)
        for local_f in range(num_cell_facets):
            entity_local_index[0] = local_f
            A10_f.fill(0.0)

            kernel_00_facet(ffi.from_buffer(A00),
                            ffi.from_buffer(null64),
                            ffi.from_buffer(null64),
                            ffi.from_buffer(coords),
                            ffi.from_buffer(entity_local_index),
                            ffi.from_buffer(null8))
            kernel_10(ffi.from_buffer(A10_f),
                      ffi.from_buffer(null64),
                      ffi.from_buffer(null64),
                      ffi.from_buffer(coords),
                      ffi.from_buffer(entity_local_index),
                      ffi.from_buffer(null8))
            start_row = local_f * Vbar_ele_space_dim
            end_row = start_row + Vbar_ele_space_dim
            A10[start_row:end_row, :] += A10_f[:, :]
        return A00, A10

    @numba.njit(fastmath=True)
    def sink(*args):
        pass

    @numba.njit(fastmath=True)
    def assemble_matrix_hdg_numba(mat, x_dofs, x, dofmap, num_cells, bc_dofs_marker, mat_set, mode):
        # TODO BCs
        coords = np.zeros((num_dofs_g, 3))
        A_local = np.zeros((num_cell_facets * Vbar_ele_space_dim,
                            num_cell_facets * Vbar_ele_space_dim),
                           dtype=PETSc.ScalarType)
        A_local_f = np.zeros((Vbar_ele_space_dim, Vbar_ele_space_dim),
                             dtype=PETSc.ScalarType)
        for cell in range(num_cells):
            for j in range(num_dofs_g):
                coords[j] = x[x_dofs[cell, j], :]
            A_local.fill(0.0)

            # FIXME Seems silly to assemble this together only to split
            # again later
            A00, A10 = compute_A00_A10(coords)
            A_local += compute_A11(coords) - A10 @ np.linalg.solve(A00, A10.T)

            cell_facets = c_to_facet_mesh_f[cell]
            for local_facet_0, facet_0 in enumerate(cell_facets):
                for local_facet_1, facet_1 in enumerate(cell_facets):
                    dofs_0 = dofmap[facet_0, :]
                    dofs_1 = dofmap[facet_1, :]

                    # FIXME Can this be done without copying?
                    A_local_f.fill(0.0)
                    A_local_f += A_local[
                        local_facet_0 * Vbar_ele_space_dim:
                        local_facet_0 * Vbar_ele_space_dim + Vbar_ele_space_dim,
                        local_facet_1 * Vbar_ele_space_dim:
                        local_facet_1 * Vbar_ele_space_dim + Vbar_ele_space_dim]

                    # FIXME Need to add block size
                    for i, dof in enumerate(dofs_0):
                        if bc_dofs_marker[dof]:
                            A_local_f[i, :] = 0.0

                    for i, dof in enumerate(dofs_1):
                        if bc_dofs_marker[dof]:
                            A_local_f[:, i] = 0.0

                    mat_set(mat, Vbar_ele_space_dim, ffi.from_buffer(dofs_0),
                            Vbar_ele_space_dim, ffi.from_buffer(dofs_1),
                            ffi.from_buffer(A_local_f), mode)

        sink(A_local, A_local_f)

    @numba.njit(fastmath=True)
    def assemble_vector_hdg(b, x_dofs, x, dofmap, num_cells):
        coords = np.zeros((num_dofs_g, 3))
        b_local = np.zeros(num_cell_facets * Vbar_ele_space_dim,
                           dtype=PETSc.ScalarType)
        b_0 = np.zeros(V_ele_space_dim, dtype=PETSc.ScalarType)
        b_local_f = np.zeros(Vbar_ele_space_dim, dtype=PETSc.ScalarType)
        for cell in range(num_cells):
            for j in range(num_dofs_g):
                coords[j] = x[x_dofs[cell, j], :]
            b_local.fill(0.0)

            b_0.fill(0.0)
            kernel_0(ffi.from_buffer(b_0),
                     ffi.from_buffer(null64),
                     ffi.from_buffer(null64),
                     ffi.from_buffer(coords),
                     ffi.from_buffer(null32),
                     ffi.from_buffer(null8))

            A00, A10 = compute_A00_A10(coords)
            b_local -= A10 @ np.linalg.solve(A00, b_0)

            cell_facets = c_to_facet_mesh_f[cell]
            for local_facet, facet in enumerate(cell_facets):
                dofs = dofmap[facet, :]

                b_local_f.fill(0.0)
                b_local_f += b_local[
                    local_facet * Vbar_ele_space_dim:
                    local_facet * Vbar_ele_space_dim + Vbar_ele_space_dim]
                b[dofs] += b_local_f
        sink(b_local, b_local_f)

    c_signature = numba.types.void(
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.int32),
        numba.types.CPointer(numba.types.uint8))

    @numba.cfunc(c_signature, nopython=True, fastmath=True)
    def tabulate_tensor(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
        A_local = numba.carray(A_, (num_cell_facets * Vbar_ele_space_dim,
                                    num_cell_facets * Vbar_ele_space_dim),
                               dtype=PETSc.ScalarType)
        coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
        A00, A10 = compute_A00_A10(coords)
        A_local += compute_A11(coords) - A10 @ np.linalg.solve(A00, A10.T)

    @numba.cfunc(c_signature, nopython=True, fastmath=True)
    def backsub(x_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
        x = numba.carray(x_, V_ele_space_dim, dtype=PETSc.ScalarType)
        coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
        u_bar = numba.carray(w_, Vbar_ele_space_dim * num_cell_facets,
                             dtype=PETSc.ScalarType)
        b_0 = np.zeros(V_ele_space_dim, dtype=PETSc.ScalarType)
        # TODO Pass nullptr for last two parameters
        kernel_0(ffi.from_buffer(b_0),
                 ffi.from_buffer(null64),
                 ffi.from_buffer(null64),
                 ffi.from_buffer(coords),
                 ffi.from_buffer(null32),
                 ffi.from_buffer(null8))
        A00, A10 = compute_A00_A10(coords)

        x += np.linalg.solve(A00, b_0 - A10.T @ u_bar)

    par_print("BC")
    with Timer("BC") as t:
        msh_boundary_facets = mesh.locate_entities_boundary(
            msh, fdim, boundary)
        facet_mesh_boundary_facets = [inv_entity_map[facet]
                                      for facet in msh_boundary_facets]
        bc_dofs = fem.locate_dofs_topological(
            Vbar, fdim, facet_mesh_boundary_facets)
        num_dofs_Vbar = (Vbar.dofmap.index_map.size_local +
                         Vbar.dofmap.index_map.num_ghosts) * V.dofmap.index_map_bs
        bc_dofs_marker = np.full(num_dofs_Vbar, False, dtype=np.bool8)
        bc_dofs_marker[bc_dofs] = True
        bc = fem.dirichletbc(PETSc.ScalarType(0.0), bc_dofs, Vbar)

    integrals = {fem.IntegralType.cell: {-1: (tabulate_tensor.address, [])}}
    # HACK: Pass empty entity maps to prevent Form complaining about different
    # meshes
    a = Form_float64(
        [Vbar._cpp_object, Vbar._cpp_object], integrals, [], [], False, msh,
        entity_maps={facet_mesh: inv_entity_map})
    # NOTE Currently this only creates sparsity
    A = assemble_matrix_hdg(a, bcs=[bc])
    A.assemble()

    par_print("Assemble vec")
    with Timer("Assemble vec") as t:
        b_func = fem.Function(Vbar)

        assemble_vector_hdg(b_func.x.array, x_dofs, x,
                            Vbar_dofmap, num_owned_cells)
        b = b_func.vector
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])

    par_print("Setup solver")
    with Timer("Setup solver") as t:
        use_direct_solver = False

        if use_direct_solver:
            ksp = PETSc.KSP().create(msh.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("superlu_dist")
        else:
            opts = PETSc.Options()
            # Solver (Use conjugate gradient)
            opts['ksp_type'] = 'cg'
            opts['ksp_rtol'] = '1.0e-8'
            opts['ksp_view'] = None
            opts['ksp_monitor'] = None
            opts['pc_type'] = 'hypre'
            opts['pc_hypre_type'] = 'boomeramg'
            opts['pc_hypre_boomeramg_strong_threshold'] = 0.7
            opts['pc_hypre_boomeramg_agg_nl'] = 4
            opts['pc_hypre_boomeramg_agg_num_paths'] = 2

            PETSc.Options().view()

            ksp = PETSc.KSP().create(msh.comm)
            ksp.setOperators(A)
            ksp.setFromOptions()

    par_print("Solve")
    with Timer("Solve") as t:
        # Compute solution
        ubar = fem.Function(Vbar)
        import time
        start = time.time()
        ksp.solve(b, ubar.vector)
        end = time.time()
        par_print(f"solve time = {end - start}")
        ubar.x.scatter_forward()

    # par_print("Write")

    # with io.VTXWriter(msh.comm, "ubar.bp", ubar) as f:
    #     f.write(0.0)

    par_print("Create form")
    with Timer("Create form") as t:
        # TODO Check with custom integration entities that this actually runs
        # over all cells
        integrals = {fem.IntegralType.cell: {-1: (backsub.address, [])}}
        u_form = Form_float64([V._cpp_object], integrals,
                              [], [], False, None, {})

    u = fem.Function(V)

    # TODO Use numba
    par_print("Pack coeffs")
    with Timer("Pack coeffs") as t:
        coeffs = np.zeros((num_cells, Vbar_ele_space_dim * num_cell_facets))
        for cell in range(num_cells):
            cell_facets = c_to_facet_mesh_f[cell]
            for local_facet, facet in enumerate(cell_facets):
                dofs = Vbar_dofmap[facet]
                coeffs[
                    cell,
                    Vbar_ele_space_dim * local_facet:
                    Vbar_ele_space_dim * local_facet + Vbar_ele_space_dim] = \
                    ubar.x.array[dofs]
        coeffs = {(IntegralType.cell, -1): coeffs}

    par_print("Assemble vec")
    with Timer("Assemble vec") as t:
        fem.assemble_vector(u.x.array, u_form, coeffs=coeffs)
        u.vector.ghostUpdate(addv=PETSc.InsertMode.ADD,
                             mode=PETSc.ScatterMode.REVERSE)

    # par_print("Write")
    # with io.VTXWriter(msh.comm, "u.bp", u) as f:
    #     f.write(0.0)

    par_print("Compute error")
    with Timer("Compute error") as t:
        e_L2 = norm_L2(msh.comm, u - u_e)
    if msh.comm.rank == 0:
        print(f"e_L2 = {e_L2}")

    par_print(f"total num dofs V = {V.dofmap.index_map.size_global}")
    par_print(f"total num dofs Vbar = {Vbar.dofmap.index_map.size_global}")
    par_print(
        f"total dofs = {V.dofmap.index_map.size_global + Vbar.dofmap.index_map.size_global}")
    par_print(
        f"total dofs per process = {(V.dofmap.index_map.size_global + Vbar.dofmap.index_map.size_global) / comm.size}")


if __name__ == "__main__":
    # import cProfile
    # cProfile.run(
    #     "main()", filename=f"out_python_{MPI.COMM_WORLD.rank}.profile")
    with Timer("TOTAL") as t:
        main()
    # list_timings(MPI.COMM_WORLD, [TimingType.wall, TimingType.user])
