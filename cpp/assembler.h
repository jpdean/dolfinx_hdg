#pragma once

#include <dolfinx/fem/Form.h>
#include <functional>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <iostream>
// #include "assemble_matrix_impl.h"
// #include "assemble_vector_impl.h"
#include <vector>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/Constant.h>
#include <cstdint>

namespace dolfinx_hdg::fem
{
    // template <typename T>
    // void assemble_vector(xtl::span<T> b,
    //                      const dolfinx::fem::Form<PetscScalar> &L,
    //                      const dolfinx::mesh::Mesh& mesh,
    //                      const dolfinx::mesh::Mesh& facet_mesh,
    //                      const xtl::span<const T> &constants,
    //                      const std::map<std::pair<dolfinx::fem::IntegralType, int>,
    //                         std::pair<xtl::span<const T>, int>>& coefficients)
    // {
    //     impl::assemble_vector(b, L, mesh, facet_mesh, constants, coefficients);
    // }

    // template <typename T>
    // void assemble_vector(xtl::span<T> b,
    //                      const dolfinx::fem::Form<PetscScalar> &L)
    // {
    //     // FIXME Think about the best way to do this. Currently, this only
    //     // packs constants / coefficients for the facet space form and these
    //     // are accessed incorrectly later
    //     const std::vector<T> constants =
    //         dolfinx::fem::pack_constants(L);
    //     const auto coefficients = dolfinx::fem::pack_coefficients(L);
    //     assemble_vector(b, L, tcb::make_span(constants),
    //                     dolfinx::fem::make_coefficients_span(coefficients));
    // }

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
        // impl::assemble_matrix(mat_add, a, constants, coefficients, dof_marker0,
        //                       dof_marker1);
        std::cout << "assemble_matrix cpp\n";
    }

    // template <typename T>
    // void assemble_matrix(
    //     const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
    //                             const std::int32_t *, const T *)> &mat_add,
    //     const dolfinx::fem::Form<T> &a,
    //     const dolfinx::mesh::Mesh& mesh,
    //     const dolfinx::mesh::Mesh& facet_mesh,
    //     const xtl::span<const T> &constants,
    //     const std::map<std::pair<dolfinx::fem::IntegralType, int>,
    //                std::pair<xtl::span<const T>, int>>& coefficients,
    //     const std::vector<
    //         std::shared_ptr<const dolfinx::fem::DirichletBC<T>>> &bcs)
    // {

    //     // Index maps for dof ranges
    //     auto map0 = a.function_spaces().at(0)->dofmap()->index_map;
    //     auto map1 = a.function_spaces().at(1)->dofmap()->index_map;
    //     auto bs0 = a.function_spaces().at(0)->dofmap()->index_map_bs();
    //     auto bs1 = a.function_spaces().at(1)->dofmap()->index_map_bs();

    //     // Build dof markers
    //     std::vector<std::int8_t> dof_marker0, dof_marker1;
    //     assert(map0);
    //     std::int32_t dim0 = bs0 * (map0->size_local() + map0->num_ghosts());
    //     assert(map1);
    //     std::int32_t dim1 = bs1 * (map1->size_local() + map1->num_ghosts());
    //     for (std::size_t k = 0; k < bcs.size(); ++k)
    //     {
    //         assert(bcs[k]);
    //         assert(bcs[k]->function_space());
    //         if (a.function_spaces().at(0)->contains(*bcs[k]->function_space()))
    //         {
    //             dof_marker0.resize(dim0, false);
    //             bcs[k]->mark_dofs(dof_marker0);
    //         }

    //         if (a.function_spaces().at(1)->contains(*bcs[k]->function_space()))
    //         {
    //             dof_marker1.resize(dim1, false);
    //             bcs[k]->mark_dofs(dof_marker1);
    //         }
    //     }

    //     // Assemble
    //     impl::assemble_matrix(mat_add, a, mesh, facet_mesh, constants, coefficients,
    //                           dof_marker0, dof_marker1);
    // }

    // template <typename T>
    // void assemble_matrix(
    //     const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
    //                             const std::int32_t *, const T *)> &mat_add,
    //     const dolfinx::fem::Form<T> &a,
    //     const std::vector<std::shared_ptr<
    //         const dolfinx::fem::DirichletBC<T>>> &bcs)
    // {
    //     // FIXME Think about the best way to do this. Currently, this only
    //     // packs constants / coefficients for the facet space form and these
    //     // are accessed incorrectly later
    //     // FIXME Is the shared pointer needed for a? Before, a was passed
    //     // to pack constants / coefficients without the dereference op.
    //     // Prepare constants and coefficients
    //     const std::vector<T> constants =
    //         dolfinx::fem::pack_constants(a);
    //     const auto coefficients =
    //         dolfinx::fem::pack_coefficients(a);

    //     // Assemble
    //     assemble_matrix(mat_add, a, tcb::make_span(constants),
    //                     dolfinx::fem::make_coefficients_span(coefficients), bcs);
    // }
}
