#include "petsc.h"
#include <iostream>

// Mat dolfinx_hdg::fem::create_matrix(const Form<PetscScalar>& a,
//                                     const std::string& type)
// {
//   // Build sparsitypattern
//   la::SparsityPattern pattern = fem::create_sparsity_pattern(a);

//   // Finalise communication
//   pattern.assemble();

//   return la::create_petsc_matrix(a.mesh()->mpi_comm(), pattern, type);
// }

void dolfinx_hdg::fem::create_matrix()
{
    std::cout << "dolfinx_hdg::fem::create_matrix\n";
}
