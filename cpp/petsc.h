#pragma once

#include <dolfinx/fem/Form.h>
#include <petscmat.h>

namespace dolfinx_hdg::fem
{
    // Mat create_matrix(const Form<PetscScalar> &a,
    Mat create_matrix(const dolfinx::fem::Form<PetscScalar> &a,
                       const std::string &type);
}
