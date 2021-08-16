#pragma once

#include <dolfinx/fem/Form.h>
#include <functional>
#include <xtl/xspan.hpp>
#include <dolfinx/common/array2d.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/utils.h>
#include <iostream>

namespace dolfinx_hdg::fem
{
    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_add,
        const dolfinx::fem::Form<T> &a,
        const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> &bcs)
    {
        // Prepare constants and coefficients
        // const std::vector<T> constants = dolfinx::fem::pack_constants(a);
        // const dolfinx::array2d<T> coeffs = dolfinx::fem::pack_coefficients(a);
        
        std::cout << "dolfinx_hdg::fem::assemble_matrix\n";
        // Assemble
        // assemble_matrix(mat_add, a, tcb::make_span(constants), coeffs, bcs);
    }

}
