from dolfinx.cpp.mesh import cell_num_entities
import numpy as np
from dolfinx import fem
import ufl
from mpi4py import MPI
from petsc4py import PETSc


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


def norm_L2(comm, v, measure=ufl.dx):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(
        fem.form(ufl.inner(v, v) * measure)), op=MPI.SUM))


def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(
            fem.Constant(msh, PETSc.ScalarType(1.0)) * ufl.dx)), op=MPI.SUM)
    return 1 / vol * msh.comm.allreduce(
        fem.assemble_scalar(fem.form(v * ufl.dx)), op=MPI.SUM)


def normal_jump_error(msh, v):
    n = ufl.FacetNormal(msh)
    return norm_L2(msh.comm, ufl.jump(v, n), measure=ufl.dS)
