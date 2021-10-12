#pragma once

// FIXME See which of these aren't needed
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/Constant.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <memory>
#include <vector>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xtensor.hpp>
#include <iostream>

namespace dolfinx_hdg::fem::impl
{
    /// Execute kernel over cells and accumulate result in vector
    /// @tparam T The scalar type
    /// @tparam _bs The block size of the form test function dof map. If
    /// less than zero the block size is determined at runtime. If `_bs` is
    /// positive the block size is used as a compile-time constant, which
    /// has performance benefits.
    template <typename T, int _bs = -1>
    void assemble_cells(
        xtl::span<T> b,
        const dolfinx::mesh::Mesh &cell_mesh,
        const dolfinx::mesh::Mesh &facet_mesh,
        const xtl::span<const std::int32_t> &cells,
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap, const int bs,
        const std::function<void(T *, const T *, const T *, const double *, const int *,
                                 const std::uint8_t *)> &fn,
        const xtl::span<const T> &constants, const xtl::span<const T> &coeffs,
        int cstride,
        const std::function<std::uint8_t(std::size_t)> &get_perm)
    {
        assert(_bs < 0 or _bs == bs);

        const int tdim = cell_mesh.topology().dim();

        // Prepare cell geometry
        const dolfinx::graph::AdjacencyList<std::int32_t> &x_dofmap =
            cell_mesh.geometry().dofmap();

        // FIXME: Add proper interface for num coordinate dofs
        const std::size_t num_dofs_g = x_dofmap.num_links(0);
        const xt::xtensor<double, 2> &x_g = cell_mesh.geometry().x();

        // FIXME: Find better way to handle facet geom
        const std::size_t facet_num_dofs_g =
            facet_mesh.geometry().dofmap().num_links(0);

        const int num_cell_facets = dolfinx::mesh::cell_num_entities(cell_mesh.topology().cell_type(), tdim - 1);

        // FIXME: Add proper interface for num_dofs
        // Create data structures used in assembly
        const int num_dofs = dofmap.links(0).size();
        std::vector<double> coordinate_dofs(3 * (num_dofs_g + num_cell_facets * facet_num_dofs_g));
        std::vector<std::uint8_t> cell_facet_perms(num_cell_facets);
        auto c_to_f = cell_mesh.topology().connectivity(tdim, tdim - 1);

        xt::xarray<T> be_sc = xt::zeros<T>({bs * num_dofs * num_cell_facets});

        for (auto cell : cells)
        {
            auto cell_facets = c_to_f->links(cell);
            auto ent_to_geom = dolfinx::mesh::entities_to_geometry(
                cell_mesh, tdim - 1, cell_facets, false);
	
            dolfinx_hdg::fem::impl_helpers::get_coordinate_dofs(
                coordinate_dofs, cell, cell_facets, x_dofmap, x_g, ent_to_geom);

            dolfinx_hdg::fem::impl_helpers::get_cell_facet_perms(
                cell_facet_perms, cell, num_cell_facets, get_perm);

            std::fill(be_sc.begin(), be_sc.end(), 0);
            fn(be_sc.data(), coeffs.data() + cell * cstride, constants.data(),
               coordinate_dofs.data(), nullptr, cell_facet_perms.data());

            for (int local_f = 0; local_f < num_cell_facets; ++local_f)
            {
                const int f = cell_facets[local_f];

                auto dofs = dofmap.links(f);

                // Vector corresponding to dofs of facets f
                xt::xarray<double> be_sc_f =
                    xt::view(be_sc,
                             xt::range(local_f * bs * num_dofs,
                                       (local_f + 1) * bs * num_dofs));

                for (int i = 0; i < num_dofs; ++i)
                    for (int k = 0; k < bs; ++k)
                        b[bs * dofs[i] + k] += be_sc_f[bs * i + k];
            }
        }
    }

    template <typename T>
    void assemble_vector(xtl::span<T> b,
                         const dolfinx::fem::Form<PetscScalar> &L,
                         const xtl::span<const T> &constants,
                         const xtl::span<const T> &coeffs,
                         int cstride)
    {
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = L.mesh();
        assert(mesh);

        std::shared_ptr<const dolfinx::mesh::Mesh> facet_mesh =
            L.function_spaces().at(0)->mesh();
        assert(facet_mesh);

        // Get dofmap data
        assert(L.function_spaces().at(0));
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap = L.function_spaces().at(0)->dofmap();
        assert(dofmap);
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofs = dofmap->list();
        const int bs = dofmap->bs();

        // TODO dof transformations and facet permutations
        std::function<std::uint8_t(std::size_t)> get_perm =
            [](std::size_t)
        { return 0; };

        for (int i : L.integral_ids(dolfinx::fem::IntegralType::cell))
        {
            const auto &fn = L.kernel(dolfinx::fem::IntegralType::cell, i);
            const std::vector<std::int32_t> &cells = L.cell_domains(i);

            const int tdim = mesh->topology().dim();
            mesh->topology_mutable().create_connectivity(tdim, tdim - 1);

            if (bs == 1)
            {
                impl::assemble_cells<T, 1>(b, *mesh, *facet_mesh, cells, dofs, bs, fn,
                                           constants, coeffs, cstride,
                                           get_perm);
            }
            else if (bs == 3)
            {
                impl::assemble_cells<T, 3>(b, *mesh, *facet_mesh, cells, dofs, bs, fn,
                                           constants, coeffs, cstride,
                                           get_perm);
            }
            else
            {
                impl::assemble_cells(b, *mesh, *facet_mesh, cells, dofs, bs, fn,
                                     constants, coeffs, cstride,
                                     get_perm);
            }
        }

        //     const int tdim = mesh->topology().dim();

        // assert(L.function_spaces().at(0));
        // const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap =
        //     L.function_spaces().at(0)->dofmap()->list();
        //     assert(dofmap);
        //     const int bs = L.function_spaces().at(0)->dofmap()->bs();
        //     const int num_dofs = dofmap.links(0).size();
        //     const int ndim = bs * num_dofs;

        //     const int codim = L.function_spaces().at(0)->codimension();

        //     // TODO DOF Transformations

        //     // TODO Is this needed?
        //     mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
        //     auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
        //     assert(c_to_f);
        //     mesh->topology_mutable().create_entity_permutations();
        //     const std::vector<std::uint8_t> &perms =
        //         mesh->topology().get_facet_permutations();

        //     // FIXME Do this properly
        //     const auto &kernel =
        //         L.kernel(dolfinx::fem::IntegralType::cell, -1);

        //     for (int c = 0; c < c_to_f->num_nodes(); ++c)
        //     {
        //         auto cell_facets = c_to_f->links(c);
        //         const int num_facets = cell_facets.size();

        //         auto coordinate_dofs = dolfinx_hdg::sc::get_cell_coord_dofs(mesh, c);

        //         // TODO Do this properly
        //         // Check codim either 0 or 1
        //         const int be_size = codim == 0 ? ndim : ndim * num_facets;
        //         xt::xarray<double> be = xt::zeros<double>({be_size});

        //         std::vector<unsigned char> cell_facet_perms(num_facets);
        //         // FIXME This assumes codim = 1. Need to generalize.
        //         std::vector<T> cell_coeffs;
        //         for (int local_f = 0; local_f < num_facets; ++local_f)
        //         {
        //             cell_facet_perms[local_f] =
        //                 perms[c * num_facets + local_f];

        //             // Pack coefficients for each of the cells facets into one
        //             // array
        //             // FIXME Do this without loop
        //             for (auto coeff : coeffs.row(cell_facets[local_f]))
        //             {
        //                 cell_coeffs.push_back(coeff);
        //             }
        //         }

        //         kernel(be.data(), cell_coeffs.data(), constants.data(),
        //                coordinate_dofs.data(), nullptr,
        //                cell_facet_perms.data());

        //         if (codim == 0)
        //         {
        //             auto dofs = dofmap.links(c);
        //             for (int i = 0; i < num_dofs; ++i)
        //                 for (int k = 0; k < bs; ++k)
        //                     b[bs * dofs[i] + k] += be[bs * i + k];
        //         }
        //         else
        //         {
        //             for (int local_f = 0; local_f < cell_facets.size(); ++local_f)
        //             {
        //                 const int f = cell_facets[local_f];

        //                 auto dofs = dofmap.links(f);

        //                 // Vector corresponding to dofs of facets f
        //                 // NOTE Have to cast to xt::xarray<double> (can't just use auto)
        //                 // otherwise it returns a view and Le_sc_f.data() gets the
        //                 // wrong values (the values) from the full Ae_sc array
        //                 xt::xarray<double> be_sc_f =
        //                     xt::view(be,
        //                              xt::range(local_f * ndim,
        //                                        local_f * ndim + ndim));

        //                 for (int i = 0; i < num_dofs; ++i)
        //                     for (int k = 0; k < bs; ++k)
        //                         b[bs * dofs[i] + k] += be_sc_f[bs * i + k];
        //             }
        //         }
        //     }
    }
}
