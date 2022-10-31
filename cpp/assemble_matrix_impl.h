#pragma once

// FIXME See which of these aren't needed
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
// #include <dolfinx/mesh/utils.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <iterator>
#include <vector>
#include <span>
#include <iostream>
// #include <xtl/xspan.hpp>
// #include <xtensor/xarray.hpp>
// #include <xtensor/xio.hpp> // To cout xtensor arrays
// #include "assembler_helpers.h"

namespace dolfinx_hdg::fem::impl
{
    // template <typename T>
    // void assemble_cells(
    //     const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
    //                             const std::int32_t*, const T*)>& mat_set,
    //     const dolfinx::mesh::Mesh& cell_mesh,
    //     const xtl::span<const std::int32_t>& cells,
    //     const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
    //     const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
    //     const xtl::span<const std::int8_t>& bc0,
    //     const xtl::span<const std::int8_t>& bc1,
    //     const std::function<void(T*, const T*, const T*, const double*, const int*,
    //                             const std::uint8_t*)>& kernel,
    //     const xtl::span<const T>& coeffs, int cstride,
    //     const xtl::span<const T>& constants,
    //     const std::function<std::uint8_t(std::size_t)>& get_perm,
    //     const std::function<std::uint8_t(std::size_t)>& get_full_cell_perm)
    // {
    //     const int tdim = cell_mesh.topology().dim();

    //     // Prepare cell geometry
    //     const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap =
    //         cell_mesh.geometry().dofmap();

    //     // FIXME: Add proper interface for num coordinate dofs
    //     const std::size_t num_dofs_g = x_dofmap.num_links(0);
    //     xtl::span<const double> x_g = cell_mesh.geometry().x();

    //     // FIXME: Find nicer way to do this
    //     const std::size_t facet_num_dofs_g =
    //         dolfinx::mesh::entities_to_geometry(cell_mesh, tdim - 1, std::vector{0}, false).shape()[1];

    //     const int num_cell_facets
    //         = dolfinx::mesh::cell_num_entities(cell_mesh.topology().cell_type(), tdim - 1);

    //     // Data structures used in assembly
    //     std::vector<double> coordinate_dofs(3 * (num_dofs_g + num_cell_facets * facet_num_dofs_g));
    //     const int num_dofs0 = dofmap0.links(0).size();
    //     const int num_dofs1 = dofmap1.links(0).size();
    //     const int ndim0 = bs0 * num_dofs0;
    //     const int ndim1 = bs1 * num_dofs1;
    //     // Each facet needs two permutations
    //     std::vector<std::uint8_t> cell_facet_perms(2 * num_cell_facets);
    //     auto c_to_f = cell_mesh.topology().connectivity(tdim, tdim - 1);

    //     xt::xarray<T> Ae_sc = xt::zeros<T>({ndim0 * num_cell_facets,
    //                                         ndim1 * num_cell_facets});

    //     for (std::size_t index = 0; index < cells.size(); ++index)
    //     {
    //         std::int32_t cell = cells[index];
    //         auto cell_facets = c_to_f->links(cell);
    //         auto ent_to_geom = dolfinx::mesh::entities_to_geometry(
    //             cell_mesh, tdim - 1, cell_facets, false);

    //         dolfinx_hdg::fem::impl_helpers::get_coordinate_dofs(
    //             coordinate_dofs, cell, cell_facets, x_dofmap, x_g, ent_to_geom);

    //         dolfinx_hdg::fem::impl_helpers::get_cell_facet_perms(
    //             cell_facet_perms, cell, cell_facets, get_perm, get_full_cell_perm);

    //         std::fill(Ae_sc.begin(), Ae_sc.end(), 0);
    //         kernel(Ae_sc.data(), coeffs.data() + index * cstride, constants.data(),
    //                coordinate_dofs.data(), nullptr, cell_facet_perms.data());

    //         // Double loop over cell facets to assemble
    //         // FIXME Is there a better way?
    //         for (int local_f_i = 0; local_f_i < num_cell_facets; ++local_f_i)
    //         {
    //             for (int local_f_j = 0; local_f_j < num_cell_facets; ++local_f_j)
    //             {
    //                 const int f_i = cell_facets[local_f_i];
    //                 const int f_j = cell_facets[local_f_j];

    //                 auto dofs0 = dofmap0.links(f_i);
    //                 auto dofs1 = dofmap1.links(f_j);

    //                 // Matrix corresponding to dofs of facets f_i and f_j
    //                 // NOTE Have to cast to xt::xarray<double> (can't just use auto)
    //                 // otherwise it returns a view and Ae_sc_f_ij.data() gets the
    //                 // wrong values (the values) from the full Ae_sc array
    //                 xt::xarray<double> Ae_sc_f_ij =
    //                     xt::view(Ae_sc,
    //                              xt::range(local_f_i * ndim0,
    //                                        local_f_i * ndim0 + ndim0),
    //                              xt::range(local_f_j * ndim1,
    //                                        local_f_j * ndim1 + ndim1));

    //                 // FIXME Might be better/more efficient to do this to Ae_sc
    //                 // directly.
    //                 if (!bc0.empty())
    //                 {
    //                     for (int i = 0; i < num_dofs0; ++i)
    //                     {
    //                         for (int k = 0; k < bs0; ++k)
    //                         {
    //                             if (bc0[bs0 * dofs0[i] + k])
    //                             {
    //                                 // Zero row bs0 * i + k
    //                                 const int row = bs0 * i + k;
    //                                 xt::view(Ae_sc_f_ij, row, xt::all()) = 0.0;
    //                             }
    //                         }
    //                     }
    //                 }

    //                 if (!bc1.empty())
    //                 {
    //                     for (int j = 0; j < num_dofs1; ++j)
    //                     {
    //                         for (int k = 0; k < bs1; ++k)
    //                         {
    //                             if (bc1[bs1 * dofs1[j] + k])
    //                             {
    //                                 // Zero column bs1 * j + k
    //                                 const int col = bs1 * j + k;
    //                                 xt::view(Ae_sc_f_ij, xt::all(), col) = 0.0;
    //                             }
    //                         }
    //                     }
    //                 }

    //                 mat_set(dofs0.size(), dofs0.data(),
    //                         dofs1.size(), dofs1.data(),
    //                         Ae_sc_f_ij.data());
    //             }
    //         }
    //     }
    // }

    // template <typename T>
    // void assemble_matrix(
    //     const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
    //                             const std::int32_t *, const T *)> &mat_set,
    //     const dolfinx::fem::Form<T> &a,
    //     const dolfinx::mesh::Mesh& mesh,
    //     const dolfinx::mesh::Mesh& facet_mesh,
    //     const xtl::span<const T> &constants,
    //     const std::map<std::pair<dolfinx::fem::IntegralType, int>,
    //                std::pair<xtl::span<const T>, int>>& coefficients,
    //     const xtl::span<const std::int8_t>& bc0,
    //     const xtl::span<const std::int8_t>& bc1)
    // {
    //     // FIXME Vector elements (i.e. block size \neq 1) might break some of this

    //     // Get dofmap data
    //     std::shared_ptr<const dolfinx::fem::DofMap> dofmap0
    //         = a.function_spaces().at(0)->dofmap();
    //     std::shared_ptr<const dolfinx::fem::DofMap> dofmap1
    //         = a.function_spaces().at(1)->dofmap();
    //     assert(dofmap0);
    //     assert(dofmap1);
    //     const dolfinx::graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
    //     const int bs0 = dofmap0->bs();
    //     const dolfinx::graph::AdjacencyList<std::int32_t>& dofs1 = dofmap1->list();
    //     const int bs1 = dofmap1->bs();

    //     // TODO dof transformations and facet permutations
    //     std::function<std::uint8_t(std::size_t)> get_perm;
    //     std::function<std::uint8_t(std::size_t)> get_full_cell_perm;

    //     if (a.needs_facet_permutations())
    //     {
    //         mesh.topology_mutable().create_entity_permutations();
    //         const std::vector<std::uint8_t>& perms
    //             = mesh.topology().get_facet_permutations();
    //         get_perm = [&perms](std::size_t i) { return perms[i]; };

    //         facet_mesh.topology_mutable().create_full_cell_permutations();
    //         const std::vector<std::uint8_t>& full_cell_perms
    //             = facet_mesh.topology().get_full_cell_permutations();
    //         get_full_cell_perm = [&full_cell_perms](std::size_t i) { return full_cell_perms[i]; };
    //     }
    //     else
    //     {
    //         get_perm = [](std::size_t) { return 0; };
    //         get_full_cell_perm = [](std::size_t) { return 0; };
    //     }

    //     for (int i : a.integral_ids(dolfinx::fem::IntegralType::cell))
    //     {
    //         const auto& fn = a.kernel(dolfinx::fem::IntegralType::cell, i);
    //         const auto& [coeffs, cstride] = coefficients.at(
    //             {dolfinx::fem::IntegralType::cell, i});
    //         const std::vector<std::int32_t>& cells = a.cell_domains(i);
    //         const int tdim = mesh.topology().dim();
    //         mesh.topology_mutable().create_connectivity(tdim, tdim - 1);
    //         impl::assemble_cells<T>(mat_set, mesh, cells, dofs0, bs0,
    //                                 dofs1, bs1, bc0, bc1, fn, coeffs,
    //                                 cstride, constants, get_perm,
    //                                 get_full_cell_perm);
    //     }
    // }

    template <typename T, typename U>
    void assemble_cells(
        U mat_set, const mesh::Mesh &mesh,
        const std::span<const std::int32_t> &cells,
        const std::function<void(const std::span<T> &,
                                 const std::span<const std::uint32_t> &,
                                 std::int32_t, int)> &dof_transform,
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap0, int bs0,
        const std::function<void(const std::span<T> &,
                                 const std::span<const std::uint32_t> &,
                                 std::int32_t, int)> &dof_transform_to_transpose,
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap1, int bs1,
        const std::span<const std::int8_t> &bc0,
        const std::span<const std::int8_t> &bc1,
        const std::function<void(T *, const T *, const T *,
                                 const dolfinx::fem::impl::scalar_value_type_t<T> *, const int *,
                                 const std::uint8_t *)> &kernel,
        const std::span<const T> &coeffs, int cstride,
        const std::span<const T> &constants,
        const std::span<const std::uint32_t> &cell_info_0,
        const std::span<const std::uint32_t> &cell_info_1,
        const std::function<std::int32_t(const std::span<const std::int32_t> &)> &
            cell_map_0,
        const std::function<std::int32_t(const std::span<const std::int32_t> &)> &
            cell_map_1)
    {
        if (cells.empty())
            return;

        // Prepare cell geometry
        const dolfinx::mesh::Geometry &geometry = mesh.geometry();
        const graph::AdjacencyList<std::int32_t> &x_dofmap = geometry.dofmap();
        const std::size_t num_dofs_g = geometry.cmap().dim();
        std::span<const double> x_g = geometry.x();

        // Iterate over active cells
        const int num_dofs0 = dofmap0.links(0).size();
        const int num_dofs1 = dofmap1.links(0).size();
        const int ndim0 = bs0 * num_dofs0;
        const int ndim1 = bs1 * num_dofs1;
        const int num_cell_facets =
            dolfinx::mesh::cell_num_entities(mesh.topology().cell_type(),
                                             mesh.topology().dim() - 1);
        std::vector<T> Ae(ndim0 * num_cell_facets * ndim1 * num_cell_facets);
        const std::span<T> _Ae(Ae);
        std::vector<std::int32_t> dofs0(num_dofs0 * num_cell_facets);
        std::vector<std::int32_t> dofs1(num_dofs1 * num_cell_facets);
        std::vector<dolfinx::fem::impl::scalar_value_type_t<T>> coordinate_dofs(3 * num_dofs_g);

        // Iterate over active cells
        for (std::size_t index = 0; index < cells.size(); ++index)
        {
            std::int32_t c = cells[index];
            //     // Map the cell in the integration domain to the cell in the mesh
            //     // each function space is defined over
            //     std::int32_t c_0 = cell_map_0(cells.subspan(index, 1));
            //     assert(c_0 >= 0);
            //     std::int32_t c_1 = cell_map_1(cells.subspan(index, 1));
            //     assert(c_1 >= 0);

            // Get cell coordinates/geometry
            auto x_dofs = x_dofmap.links(c);
            for (std::size_t i = 0; i < x_dofs.size(); ++i)
            {
                std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                            std::next(coordinate_dofs.begin(), 3 * i));
            }

            // Tabulate tensor
            std::fill(Ae.begin(), Ae.end(), 0);
            kernel(Ae.data(), coeffs.data() + index * cstride, constants.data(),
                   coordinate_dofs.data(), nullptr, nullptr);

            // for (auto val : Ae)
            // {
            //     std::cout << val << " ";
            // }
            // std::cout << "\n";

            //     dof_transform(_Ae, cell_info_1, c_1, ndim1);
            //     dof_transform_to_transpose(_Ae, cell_info_0, c_0, ndim0);

            for (int local_facet = 0; local_facet < num_cell_facets; ++local_facet)
            {
                const std::array cell_local_facet = {c, local_facet};
                const std::int32_t facet_0 = cell_map_0(cell_local_facet);
                const std::int32_t facet_1 = cell_map_1(cell_local_facet);
                auto dofs0_f = dofmap0.links(facet_0);
                auto dofs1_f = dofmap1.links(facet_1);

                std::copy_n(dofs0_f.begin(), dofs0_f.size(), dofs0.begin() + num_dofs0 * local_facet);
                std::copy_n(dofs1_f.begin(), dofs1_f.size(), dofs1.begin() + num_dofs1 * local_facet);
            }

            // Zero rows/columns for essential bcs
            // FIXME This might be wrong for when block size is not equal to 1
            if (!bc0.empty())
            {
                for (int i = 0; i < dofs0.size(); ++i)
                {
                    for (int k = 0; k < bs0; ++k)
                    {
                        if (bc0[bs0 * dofs0[i] + k])
                        {
                            // Zero row bs0 * i + k
                            const int row = bs0 * i + k;
                            std::fill_n(std::next(Ae.begin(), num_cell_facets * ndim1 * row),
                                        num_cell_facets * ndim1, 0.0);
                        }
                    }
                }
            }

            if (!bc1.empty())
            {
                for (int j = 0; j < dofs1.size(); ++j)
                {
                    for (int k = 0; k < bs1; ++k)
                    {
                        if (bc1[bs1 * dofs1[j] + k])
                        {
                            // Zero column bs1 * j + k
                            const int col = bs1 * j + k;
                            for (int row = 0; row < num_cell_facets * ndim0; ++row)
                                Ae[row * num_cell_facets * ndim1 + col] = 0.0;
                        }
                    }
                }
            }

            mat_set(dofs0, dofs1, Ae);
        }
    }

    template <typename T, typename U>
    void assemble_matrix(
        U mat_set, const dolfinx::fem::Form<T> &a, const std::span<const T> &constants,
        const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                       std::pair<std::span<const T>, int>> &coefficients,
        const std::span<const std::int8_t> &bc0,
        const std::span<const std::int8_t> &bc1)
    {
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a.mesh();
        assert(mesh);

        // Get dofmap data
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap0 = a.function_spaces().at(0)->dofmap();
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap1 = a.function_spaces().at(1)->dofmap();
        assert(dofmap0);
        assert(dofmap1);
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofs0 = dofmap0->list();
        const int bs0 = dofmap0->bs();
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofs1 = dofmap1->list();
        const int bs1 = dofmap1->bs();

        std::shared_ptr<const dolfinx::fem::FiniteElement> element0 = a.function_spaces().at(0)->element();
        std::shared_ptr<const dolfinx::fem::FiniteElement> element1 = a.function_spaces().at(1)->element();
        const std::function<void(const std::span<T> &,
                                 const std::span<const std::uint32_t> &, std::int32_t,
                                 int)> &dof_transform = element0->get_dof_transformation_function<T>();
        const std::function<void(const std::span<T> &,
                                 const std::span<const std::uint32_t> &, std::int32_t,
                                 int)> &dof_transform_to_transpose = element1->get_dof_transformation_to_transpose_function<T>();

        const bool needs_transformation_data = element0->needs_dof_transformations() or element1->needs_dof_transformations() or a.needs_facet_permutations();
        // Get the cell infos for the meshes each function space is defined over
        // NOTE: These should only differ for mixed dimensional integrals, so
        // this could be simplified
        std::span<const std::uint32_t> cell_info_0;
        std::span<const std::uint32_t> cell_info_1;
        if (needs_transformation_data)
        {
            auto mesh_0 = a.function_spaces().at(0)->mesh();
            auto mesh_1 = a.function_spaces().at(1)->mesh();
            mesh_0->topology_mutable().create_entity_permutations();
            mesh_1->topology_mutable().create_entity_permutations();
            cell_info_0 = std::span(mesh_0->topology().get_cell_permutation_info());
            cell_info_1 = std::span(mesh_1->topology().get_cell_permutation_info());
        }

        const auto entity_map_0 = a.function_space_to_entity_map(*a.function_spaces().at(0));
        const auto entity_map_1 = a.function_space_to_entity_map(*a.function_spaces().at(1));

        for (int i : a.integral_ids(dolfinx::fem::IntegralType::cell))
        {
            const auto &fn = a.kernel(dolfinx::fem::IntegralType::cell, i);
            const auto &[coeffs, cstride] = coefficients.at({dolfinx::fem::IntegralType::cell, i});
            const std::vector<std::int32_t> &cells = a.cell_domains(i);
            impl::assemble_cells(mat_set, *mesh, cells, dof_transform, dofs0,
                                 bs0, dof_transform_to_transpose, dofs1, bs1, bc0, bc1,
                                 fn, coeffs, cstride, constants, cell_info_0,
                                 cell_info_1, entity_map_0, entity_map_1);
        }
    }
}
