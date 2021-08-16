#pragma once

#include <dolfinx/fem/Form.h>
#include <functional>
#include <xtl/xspan.hpp>
#include <dolfinx/common/array2d.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/utils.h>
#include <iostream>
#include "assemble_matrix_impl.h"
#include <vector>
#include <dolfinx/fem/Constant.h>


namespace dolfinx_hdg::fem
{
    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_add,
        const dolfinx::fem::Form<T> &a, const xtl::span<const T> &constants,
        const dolfinx::array2d<T> &coeffs,
        const std::vector<
            std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> &bcs)
    {
        // Index maps for dof ranges
        auto map0 = a.function_spaces().at(0)->dofmap()->index_map;
        auto map1 = a.function_spaces().at(1)->dofmap()->index_map;
        auto bs0 = a.function_spaces().at(0)->dofmap()->index_map_bs();
        auto bs1 = a.function_spaces().at(1)->dofmap()->index_map_bs();

        // Build dof markers
        std::vector<bool> dof_marker0, dof_marker1;
        assert(map0);
        std::int32_t dim0 = bs0 * (map0->size_local() + map0->num_ghosts());
        assert(map1);
        std::int32_t dim1 = bs1 * (map1->size_local() + map1->num_ghosts());
        for (std::size_t k = 0; k < bcs.size(); ++k)
        {
            assert(bcs[k]);
            assert(bcs[k]->function_space());
            if (a.function_spaces().at(0)->contains(*bcs[k]->function_space()))
            {
                dof_marker0.resize(dim0, false);
                bcs[k]->mark_dofs(dof_marker0);
            }

            if (a.function_spaces().at(1)->contains(*bcs[k]->function_space()))
            {
                dof_marker1.resize(dim1, false);
                bcs[k]->mark_dofs(dof_marker1);
            }
        }

        // Assemble
        impl::assemble_matrix(mat_add, a, constants, coeffs, dof_marker0,
                              dof_marker1);
    }

    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_add,
        const dolfinx::fem::Form<T> &a,
        const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> &bcs)
    {
        // Prepare constants and coefficients
        const std::vector<T> constants = dolfinx::fem::pack_constants(a);
        const dolfinx::array2d<T> coeffs = dolfinx::fem::pack_coefficients(a);

        // Assemble
        assemble_matrix(mat_add, a, tcb::make_span(constants), coeffs, bcs);
    }
}