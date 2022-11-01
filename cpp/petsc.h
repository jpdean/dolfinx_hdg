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
    Mat create_matrix(const dolfinx::fem::Form<PetscScalar> &a,
                      const std::string &type = std::string());

    /// Initialise a monolithic matrix for an array of bilinear forms
    /// @param[in] a Rectangular array of bilinear forms. The `a(i, j)` form
    /// will correspond to the `(i, j)` block in the returned matrix
    /// @param[in] type The type of PETSc Mat. If empty the PETSc default is
    /// used.
    /// @return A sparse matrix  with a layout and sparsity that matches the
    /// bilinear forms. The caller is responsible for destroying the Mat
    /// object.
    Mat create_matrix_block(
        const std::vector<std::vector<const dolfinx::fem::Form<PetscScalar> *>> &a,
        const std::string &type = std::string());
}
