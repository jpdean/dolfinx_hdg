#pragma once

#include <dolfinx/fem/Form.h>
#include <petscmat.h>

namespace dolfinx_hdg::fem
{
    /// Create a matrix
    /// @param[in] a A bilinear form
    /// @param[in] type The PETSc matrix type to create
    /// @return A sparse matrix with a layout and sparsity that matches the
    /// bilinear form. The caller is responsible for destroying the Mat
    /// object.
    // Mat create_matrix(const dolfinx::fem::Form<PetscScalar> &a,
    void create_matrix(const dolfinx::fem::Form<PetscScalar> &a,
                       const std::string &type = std::string());
}
