#pragma once

#include <dolfinx/fem/Form.h>
#include <functional>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <iostream>
#include "assemble_matrix_impl.h"
#include "assemble_vector_impl.h"
#include <vector>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/Constant.h>
#include <cstdint>

namespace dolfinx_hdg::fem
{
    /// Assemble linear form into a vector, The caller supplies the form
    /// constants and coefficients for this version, which has efficiency
    /// benefits if the data can be re-used for multiple calls.
    /// @param[in,out] b The vector to be assembled. It will not be zeroed
    /// before assembly.
    /// @param[in] L The linear forms to assemble into b
    /// @param[in] constants The constants that appear in `L`
    /// @param[in] coefficients The coefficients that appear in `L`
    template <typename T>
    void assemble_vector(
        std::span<T> b, const dolfinx::fem::Form<T> &L,
        const std::span<const T> &constants,
        const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                       std::pair<std::span<const T>, int>> &coefficients)
    {
        impl::assemble_vector(b, L, constants, coefficients);
    }

    /// Assemble bilinear form into a matrix
    /// @param[in] mat_add The function for adding values into the matrix
    /// @param[in] a The bilinear from to assemble
    /// @param[in] constants Constants that appear in `a`
    /// @param[in] coefficients Coefficients that appear in `a`
    /// @param[in] bcs Boundary conditions to apply. For boundary condition
    ///  dofs the row and column are zeroed. The diagonal  entry is not set.
    template <typename T, typename U>
    void assemble_matrix(
        U mat_add, const dolfinx::fem::Form<T> &a, const std::span<const T> &constants,
        const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                       std::pair<std::span<const T>, int>> &coefficients,
        const std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> &bcs)
    {
        // Index maps for dof ranges
        auto map0 = a.function_spaces().at(0)->dofmap()->index_map;
        auto map1 = a.function_spaces().at(1)->dofmap()->index_map;
        auto bs0 = a.function_spaces().at(0)->dofmap()->index_map_bs();
        auto bs1 = a.function_spaces().at(1)->dofmap()->index_map_bs();

        // Build dof markers
        std::vector<std::int8_t> dof_marker0, dof_marker1;
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
        impl::assemble_matrix(mat_add, a, constants, coefficients, dof_marker0,
                              dof_marker1);
    }
}
