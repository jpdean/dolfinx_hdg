#include "petsc.h"
#include <iostream>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx_hdg/utils.h>
#include <dolfinx/la/PETScMatrix.h>

// Mat dolfinx_hdg::fem::create_matrix(const Form<PetscScalar>& a,
Mat dolfinx_hdg::fem::create_matrix(const dolfinx::fem::Form<PetscScalar> &a,
                                    const std::string &type)
{
    // Build sparsitypattern
    dolfinx::la::SparsityPattern sp =
        dolfinx_hdg::fem::create_sparsity_pattern(a);

    // Finalise communication
    sp.assemble();

    return dolfinx::la::create_petsc_matrix(a.mesh()->mpi_comm(), sp, type);
}
