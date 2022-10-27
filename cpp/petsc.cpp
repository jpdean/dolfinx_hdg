#include "petsc.h"
#include <iostream>
#include <dolfinx/la/SparsityPattern.h>
#include "utils.h"
#include <dolfinx/la/petsc.h>

// Mat dolfinx_hdg::fem::create_matrix(const dolfinx::fem::Form<PetscScalar>& a,
void dolfinx_hdg::fem::create_matrix(const dolfinx::fem::Form<PetscScalar> &a,
                                     const std::string &type)
{
    //   Build sparsity pattern
    dolfinx::la::SparsityPattern pattern =
        dolfinx_hdg::fem::create_sparsity_pattern(a);

    // Finalise communication
    pattern.assemble();

    //   return la::petsc::create_matrix(a.mesh()->comm(), pattern, type);
    std::cout << "TODO dolfinx_hdg::fem::create_matrix\n";
}
