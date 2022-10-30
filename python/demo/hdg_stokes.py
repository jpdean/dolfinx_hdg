from dolfinx import mesh, fem, io
from mpi4py import MPI
from utils import reorder_mesh
from dolfinx.cpp.mesh import cell_num_entities
import numpy as np
from ufl import inner, grad, dot, div
import ufl

comm = MPI.COMM_WORLD
rank = comm.rank
out_str = f"rank {rank}:\n"

n = 2
msh = mesh.create_unit_square(
    comm, n, n, ghost_mode=mesh.GhostMode.none)

# Currently, permutations are not working in parallel, so reorder the
# mesh
reorder_mesh(msh)

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

k = 2
V = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k))
Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k - 1))
Vbar = fem.VectorFunctionSpace(
    facet_mesh, ("Discontinuous Lagrange", k))
Qbar = fem.FunctionSpace(facet_mesh, ("Discontinuous Lagrange", k))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
p = ufl.TrialFunction(Q)
q = ufl.TestFunction(Q)
ubar = ufl.TrialFunction(Vbar)
vbar = ufl.TestFunction(Vbar)
pbar = ufl.TrialFunction(Qbar)
qbar = ufl.TestFunction(Qbar)

V_ele_space_dim = V.element.space_dimension
Vbar_ele_space_dim = Vbar.element.space_dimension
Q_ele_space_dim = Q.element.space_dimension
Qbar_ele_space_dim = Qbar.element.space_dimension

num_cell_facets = msh.ufl_cell().num_facets()
num_dofs_g = len(msh.geometry.dofmap.links(0))

h = ufl.CellDiameter(msh)
n = ufl.FacetNormal(msh)
gamma = 6.0 * k**2 / h


def u_e(x):
    return ufl.as_vector(
        (x[0]**2 * (1 - x[0])**2 * (2 * x[1] - 6 * x[1]**2 + 4 * x[1]**3),
         - x[1]**2 * (1 - x[1])**2 * (2 * x[0] - 6 * x[0]**2 + 4 * x[0]**3)))


def p_e(x):
    return x[0] * (1 - x[0])


dx_c = ufl.Measure("dx", domain=msh)
ds_c = ufl.Measure("ds", domain=msh)

x = ufl.SpatialCoordinate(msh)
f = - div(grad(u_e(x))) + grad(p_e(x))

a_00 = inner(grad(u), grad(v)) * dx_c + gamma * inner(u, v) * ds_c \
    - (inner(u, dot(grad(v), n))
       + inner(v, dot(grad(u), n))) * ds_c
a_10 = - inner(q, div(u)) * dx_c
a_20 = inner(vbar, dot(grad(u), n)) * ds_c - gamma * inner(vbar, u) * ds_c
a_30 = inner(dot(u, n), qbar) * ds_c
a_22 = gamma * inner(ubar, vbar) * ds_c

L_0 = inner(f, v) * dx_c

inv_entity_map = np.full_like(entity_map, -1)
for i, f in enumerate(entity_map):
    inv_entity_map[f] = i
