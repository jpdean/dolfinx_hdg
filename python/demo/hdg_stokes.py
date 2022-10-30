from dolfinx import mesh, fem, io
from mpi4py import MPI
from utils import reorder_mesh

comm = MPI.COMM_WORLD
rank = comm.rank
out_str = f"rank {rank}:\n"

n = 2
msh = mesh.create_unit_square(
    comm, n, n, ghost_mode=mesh.GhostMode.none)

# Currently, permutations are not working in parallel, so reorder the
# mesh
reorder_mesh(msh)
