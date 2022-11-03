#pragma once

// FIXME See which of these aren't needed
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <vector>
#include <iostream>
#include <span>
#include <algorithm>

namespace dolfinx_hdg::fem::impl
{
    /// Modify b such that:
    ///
    ///   b <- b - scale * A_j (g_j - x0_j)
    ///
    /// where j is a block (nest) row index. For a non-blocked problem j = 0.
    /// The boundary conditions bc1 are on the trial spaces V_j. The forms
    /// in [a] must have the same test space as L (from which b was built),
    /// but the trial space may differ. If x0 is not supplied, then it is
    /// treated as zero.
    /// @param[in,out] b The vector to be modified
    /// @param[in] a The bilinear forms, where a[j] is the form that
    /// generates A_j
    /// @param[in] constants Constants that appear in `a`
    /// @param[in] coeffs Coefficients that appear in `a`
    /// @param[in] bcs1 List of boundary conditions for each block, i.e.
    /// bcs1[2] are the boundary conditions applied to the columns of a[2] /
    /// x0[2] block
    /// @param[in] x0 The vectors used in the lifting
    /// @param[in] scale Scaling to apply
    template <typename T>
    void apply_lifting(
        std::span<T> b, const std::vector<std::shared_ptr<const dolfinx::fem::Form<T>>> a,
        const std::vector<std::span<const T>> &constants,
        const std::vector<std::map<std::pair<dolfinx::fem::IntegralType, int>,
                                   std::pair<std::span<const T>, int>>> &coeffs,
        const std::vector<std::vector<std::shared_ptr<const dolfinx::fem::DirichletBC<T>>>> &bcs1,
        const std::vector<std::span<const T>> &x0, double scale)
    {
        // FIXME: make changes to reactivate this check
        if (!x0.empty() and x0.size() != a.size())
        {
            throw std::runtime_error(
                "Mismatch in size between x0 and bilinear form in assembler.");
        }

        if (a.size() != bcs1.size())
        {
            throw std::runtime_error(
                "Mismatch in size between a and bcs in assembler.");
        }

        for (std::size_t j = 0; j < a.size(); ++j)
        {
            std::vector<std::int8_t> bc_markers1;
            std::vector<T> bc_values1;
            if (a[j] and !bcs1[j].empty())
            {
                assert(a[j]->function_spaces().at(0));

                auto V1 = a[j]->function_spaces()[1];
                assert(V1);
                auto map1 = V1->dofmap()->index_map;
                const int bs1 = V1->dofmap()->index_map_bs();
                assert(map1);
                const int crange = bs1 * (map1->size_local() + map1->num_ghosts());
                bc_markers1.assign(crange, false);
                bc_values1.assign(crange, 0.0);
                for (const std::shared_ptr<const dolfinx::fem::DirichletBC<T>> &bc : bcs1[j])
                {
                    bc->mark_dofs(bc_markers1);
                    bc->dof_values(bc_values1);
                }

                // if (!x0.empty())
                // {
                //     lift_bc<T>(b, *a[j], constants[j], coeffs[j], bc_values1, bc_markers1,
                //                x0[j], scale);
                // }
                // else
                // {
                //     lift_bc<T>(b, *a[j], constants[j], coeffs[j], bc_values1, bc_markers1,
                //                std::span<const T>(), scale);
                // }
            }
        }
    }

    /// Execute kernel over cells and accumulate result in vector
    /// @tparam T The scalar type
    /// @tparam _bs The block size of the form test function dof map. If
    /// less than zero the block size is determined at runtime. If `_bs` is
    /// positive the block size is used as a compile-time constant, which
    /// has performance benefits.
    template <typename T, int _bs = -1>
    void assemble_cells(
        const std::function<void(const std::span<T> &,
                                 const std::span<const std::uint32_t> &,
                                 std::int32_t, int)> &dof_transform,
        std::span<T> b, const dolfinx::mesh::Mesh &mesh,
        const std::span<const std::int32_t> &cells,
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap, int bs,
        const std::function<void(T *, const T *, const T *,
                                 const dolfinx::fem::impl::scalar_value_type_t<T> *, const int *,
                                 const std::uint8_t *)> &kernel,
        const std::span<const T> &constants, const std::span<const T> &coeffs,
        int cstride, const std::span<const std::uint32_t> &cell_info,
        const std::function<std::int32_t(const std::span<const std::int32_t> &)> &
            cell_map)
    {
        assert(_bs < 0 or _bs == bs);

        if (cells.empty())
            return;

        // Prepare cell geometry
        const dolfinx::mesh::Geometry &geometry = mesh.geometry();
        const dolfinx::graph::AdjacencyList<std::int32_t> &x_dofmap = geometry.dofmap();
        const std::size_t num_dofs_g = geometry.cmap().dim();
        std::span<const double> x_g = geometry.x();

        const int num_cell_facets =
            dolfinx::mesh::cell_num_entities(mesh.topology().cell_type(),
                                             mesh.topology().dim() - 1);

        // FIXME: Add proper interface for num_dofs
        // Create data structures used in assembly
        const int num_dofs = dofmap.links(0).size();
        std::vector<dolfinx::fem::impl::scalar_value_type_t<T>> coordinate_dofs(3 * num_dofs_g);
        std::vector<T> be(bs * num_dofs * num_cell_facets);
        const std::span<T> _be(be);
        std::vector<std::int32_t> dofs(num_dofs * num_cell_facets);

        // Iterate over active cells
        for (std::size_t index = 0; index < cells.size(); ++index)
        {
            std::int32_t c = cells[index];

            // Get cell coordinates/geometry
            auto x_dofs = x_dofmap.links(c);
            for (std::size_t i = 0; i < x_dofs.size(); ++i)
            {
                std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                            std::next(coordinate_dofs.begin(), 3 * i));
            }

            // Tabulate vector for cell
            std::fill(be.begin(), be.end(), 0);
            kernel(be.data(), coeffs.data() + index * cstride, constants.data(),
                   coordinate_dofs.data(), nullptr, nullptr);
            // TODO Dof transformations
            // dof_transform(_be, cell_info, c_0, 1);

            for (int local_facet = 0; local_facet < num_cell_facets; ++local_facet)
            {
                const std::array cell_local_facet = {c, local_facet};
                const std::int32_t facet = cell_map(cell_local_facet);
                auto dofs_f = dofmap.links(facet);
                std::copy_n(dofs_f.begin(), dofs_f.size(), dofs.begin() + num_dofs * local_facet);
            }

            // Scatter cell vector to 'global' vector array
            if constexpr (_bs > 0)
            {
                // FIXME This might be incorrect for bs > 1
                for (int i = 0; i < num_dofs * num_cell_facets; ++i)
                    for (int k = 0; k < _bs; ++k)
                        b[_bs * dofs[i] + k] += be[_bs * i + k];
            }
            else
            {
                for (int i = 0; i < num_dofs * num_cell_facets; ++i)
                    for (int k = 0; k < bs; ++k)
                        b[bs * dofs[i] + k] += be[bs * i + k];
            }
        }
    }

    /// Assemble linear form into a vector
    /// @param[in,out] b The vector to be assembled. It will not be zeroed
    /// before assembly.
    /// @param[in] L The linear forms to assemble into b
    /// @param[in] constants Packed constants that appear in `L`
    /// @param[in] coefficients Packed coefficients that appear in `L`
    template <typename T>
    void assemble_vector(
        std::span<T> b, const dolfinx::fem::Form<T> &L,
        const std::span<const T> &constants,
        const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                       std::pair<std::span<const T>, int>> &coefficients)
    {
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = L.mesh();
        assert(mesh);

        // Get dofmap data
        assert(L.function_spaces().at(0));
        std::shared_ptr<const dolfinx::fem::FiniteElement> element = L.function_spaces().at(0)->element();
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap = L.function_spaces().at(0)->dofmap();
        assert(dofmap);
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofs = dofmap->list();
        const int bs = dofmap->bs();

        const std::function<void(const std::span<T> &,
                                 const std::span<const std::uint32_t> &, std::int32_t,
                                 int)>
            dof_transform = element->get_dof_transformation_function<T>();

        const bool needs_transformation_data = element->needs_dof_transformations() or L.needs_facet_permutations();
        std::span<const std::uint32_t> cell_info;
        if (needs_transformation_data)
        {
            auto mesh_0 = L.function_spaces().at(0)->mesh();
            mesh_0->topology_mutable().create_entity_permutations();
            cell_info = std::span(mesh_0->topology().get_cell_permutation_info());
        }

        const auto entity_map = L.function_space_to_entity_map(*L.function_spaces().at(0));

        for (int i : L.integral_ids(dolfinx::fem::IntegralType::cell))
        {
            const auto &fn = L.kernel(dolfinx::fem::IntegralType::cell, i);
            const auto &[coeffs, cstride] = coefficients.at({dolfinx::fem::IntegralType::cell, i});
            const std::vector<std::int32_t> &cells = L.cell_domains(i);
            if (bs == 1)
            {
                impl::assemble_cells<T, 1>(dof_transform, b, *mesh, cells,
                                           dofs, bs, fn, constants, coeffs, cstride,
                                           cell_info, entity_map);
            }
            else if (bs == 3)
            {
                impl::assemble_cells<T, 3>(dof_transform, b, *mesh, cells,
                                           dofs, bs, fn, constants, coeffs, cstride,
                                           cell_info, entity_map);
            }
            else
            {
                impl::assemble_cells(dof_transform, b, *mesh, cells, dofs, bs,
                                     fn, constants, coeffs, cstride, cell_info,
                                     entity_map);
            }
        }
    }
}
