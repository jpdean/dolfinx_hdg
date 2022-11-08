# TODO Add fastmath flag to numba (see custom assembler demos)

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
from dolfinx_hdg.assemble import assemble_vector as assemble_vector_hdg
from dolfinx_hdg.assemble import pack_coefficients
from dolfinx_hdg.assemble import apply_lifting as apply_lifting_hdg
from utils import reorder_mesh, norm_L2
import json


def main():
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

    def print_and_time(name):
        par_print(name)
        return Timer(name)

    def u_e(x, module=np):
        u_e = 1
        for i in range(tdim):
            u_e *= module.cos(module.pi * x[i])
        return u_e

    total_timer = Timer("TOTAL")
    comm = MPI.COMM_WORLD

    timings = {}
    # n = 8
    n = round((500000 * comm.size / 60)**(1 / 3))
    timer = print_and_time(f"Create mesh (n = {n})")
    # msh = mesh.create_unit_square(
    #     comm, n, n, ghost_mode=mesh.GhostMode.none)
    msh = mesh.create_unit_cube(
        comm, n, n, n, ghost_mode=mesh.GhostMode.none)
    timings["create_mesh"] = timer.stop()

    timer = print_and_time("Reorder mesh")
    reorder_mesh(msh)
    timings["reorder_mesh"] = timer.stop()

    timer = print_and_time("Create facet mesh")
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
    timings["create_facet_mesh"] = timer.stop()

    timer = print_and_time("Create function spaces")
    k = 1
    V = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
    Vbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

    V_ele_space_dim = V.element.space_dimension
    Vbar_ele_space_dim = Vbar.element.space_dimension

    num_cell_facets = msh.ufl_cell().num_facets()
    num_dofs_g = len(msh.geometry.dofmap.links(0))
    timings["create_function_spaces"] = timer.stop()

    timer = print_and_time("Define problem")
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
    f = - div(grad(u_e(x, ufl)))
    L_0 = inner(f, v) * dx_c
    timings["define_problem"] = timer.stop()

    timer = print_and_time("Create inverse entity map")
    inv_entity_map = np.full_like(entity_map, -1)
    for i, f in enumerate(entity_map):
        inv_entity_map[f] = i
    timings["create_inv_ent_map"] = timer.stop()

    timer = print_and_time("JIT kernels")
    nptype = "float64"
    ffcxtype = "double"

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

    ffi = cffi.FFI()
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

    c_signature = numba.types.void(
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
        numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.int32),
        numba.types.CPointer(numba.types.uint8))

    @numba.cfunc(c_signature, nopython=True, fastmath=True)
    def tabulate_tensor_a(A_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
        A_local = numba.carray(A_, (num_cell_facets * Vbar_ele_space_dim,
                                    num_cell_facets * Vbar_ele_space_dim),
                               dtype=PETSc.ScalarType)
        coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
        A00, A10 = compute_A00_A10(coords)
        A_local += compute_A11(coords) - A10 @ np.linalg.solve(A00, A10.T)

    @numba.cfunc(c_signature, nopython=True, fastmath=True)
    def tabulate_tensor_L(b_, w_, c_, coords_, entity_local_index, permutation=ffi.NULL):
        b_local = numba.carray(b_, num_cell_facets * Vbar_ele_space_dim,
                               dtype=PETSc.ScalarType)
        coords = numba.carray(coords_, (num_dofs_g, 3), dtype=PETSc.ScalarType)
        b_0 = np.zeros(V_ele_space_dim, dtype=PETSc.ScalarType)
        kernel_0(ffi.from_buffer(b_0),
                 ffi.from_buffer(null64),
                 ffi.from_buffer(null64),
                 ffi.from_buffer(coords),
                 ffi.from_buffer(null32),
                 ffi.from_buffer(null8))
        A00, A10 = compute_A00_A10(coords)
        b_local -= A10 @ np.linalg.solve(A00, b_0)

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

    timings["jit_kernels"] = timer.stop()

    timer = print_and_time("Boundary conditions")
    msh_boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, boundary)
    facet_mesh_boundary_facets = inv_entity_map[msh_boundary_facets]
    bc_dofs = fem.locate_dofs_topological(
        Vbar, fdim, facet_mesh_boundary_facets)
    u_bc = fem.Function(Vbar)
    u_bc.interpolate(u_e)
    bc = fem.dirichletbc(u_bc, bc_dofs)
    timings["bcs"] = timer.stop()

    timer = print_and_time("Assemble matrix")
    integrals_a = {
        fem.IntegralType.cell: {-1: (tabulate_tensor_a.address, [])}}
    a = Form_float64(
        [Vbar._cpp_object, Vbar._cpp_object], integrals_a, [], [], False, msh,
        entity_maps={facet_mesh: inv_entity_map})
    A = assemble_matrix_hdg(a, bcs=[bc])
    A.assemble()
    timings["assemble_mat"] = timer.stop()

    timer = print_and_time("Assemble vector")
    integrals_L = {
        fem.IntegralType.cell: {-1: (tabulate_tensor_L.address, [])}}
    L = Form_float64(
        [Vbar._cpp_object], integrals_L, [], [], False, msh,
        entity_maps={facet_mesh: inv_entity_map})
    b = assemble_vector_hdg(L)
    apply_lifting_hdg(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                    mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, [bc])
    timings["assemble_vec"] = timer.stop()

    timer = print_and_time("Setup solver")
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
        opts['pc_hypre_boomeramg_strong_threshold'] = 0.85
        opts['pc_hypre_boomeramg_agg_nl'] = 1
        # opts['pc_hypre_boomeramg_agg_num_paths'] = 2

        PETSc.Options().view()

        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setFromOptions()

    timings["setup_solver"] = timer.stop()

    timer = print_and_time("Solve")
    # Compute solution
    ubar = fem.Function(Vbar)
    ksp.solve(b, ubar.vector)
    ubar.x.scatter_forward()
    timings["solve"] = timer.stop()

    # par_print("Write")

    # with io.VTXWriter(msh.comm, "ubar.bp", ubar) as f:
    #     f.write(0.0)

    timer = print_and_time("Backsubstitution")
    # TODO Check with custom integration entities that this actually runs
    # over all cells
    integrals = {fem.IntegralType.cell: {-1: (backsub.address, [])}}
    u_form = Form_float64([V._cpp_object], integrals,
                            [ubar._cpp_object], [], False, None,
                            entity_maps={facet_mesh: inv_entity_map})
    coeffs = pack_coefficients(u_form)

    u = fem.Function(V)
    fem.assemble_vector(u.x.array, u_form, coeffs=coeffs)
    u.vector.ghostUpdate(addv=PETSc.InsertMode.ADD,
                            mode=PETSc.ScatterMode.REVERSE)
    timings["backsub"] = timer.stop()

    # par_print("Write")
    # with io.VTXWriter(msh.comm, "u.bp", u) as f:
    #     f.write(0.0)

    timer = print_and_time("Compute erorrs")
    e_L2 = norm_L2(msh.comm, u - u_e(x, ufl))
    timings["compute_errors_cell"] = timer.stop()

    timings["total"] = total_timer.stop()

    if msh.comm.rank == 0:
        print(f"e_L2 = {e_L2}")

    num_cells = msh.topology.index_map(tdim).size_global
    num_dofs_V = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    num_dofs_Vbar = Vbar.dofmap.index_map.size_global * Vbar.dofmap.index_map_bs
    total_dofs = num_dofs_V + num_dofs_Vbar

    data = {}
    data["num_proc"] = comm.size
    data["num_cells"] = num_cells
    data["num_dofs_V"] = num_dofs_V
    data["num_dofs_Vbar"] = num_dofs_Vbar
    data["total_dofs"] = total_dofs
    data["e_u"] = e_L2
    data["its"] = ksp.its

    results = {}
    results["data"] = data
    results["timings"] = {}
    for name, t in timings.items():
        results["timings"][name] = comm.allreduce(t, op=MPI.MAX)

    for name, val in results["data"].items():
        par_print(f"{name} = {val}")

    if comm.rank == 0:
        with open(f"results_{comm.size}.json", "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    # import cProfile
    # cProfile.run(
    #     "main()", filename=f"out_python_{MPI.COMM_WORLD.rank}.profile")
    main()
    list_timings(MPI.COMM_WORLD, [TimingType.wall, TimingType.user])
