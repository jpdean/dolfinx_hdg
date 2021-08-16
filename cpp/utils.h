#pragma once

#include <dolfinx/fem/Form.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/fem/utils.h>

namespace dolfinx_hdg::fem
{
    template <typename T>
    dolfinx::la::SparsityPattern create_sparsity_pattern(
        const dolfinx::fem::Form<T>& a)
    {
        dolfinx::la::SparsityPattern sp = dolfinx::fem::create_sparsity_pattern(a);
        sp.assemble();
        return sp;
    }
}
