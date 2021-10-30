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
#include <dolfinx/mesh/Topology.h>
#include <functional>
#include <iterator>
#include <vector>
#include <xtl/xspan.hpp>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> // To cout xtensor arrays
#include "assembler_helpers.h"

namespace dolfinx_hdg::fem::impl
{
    template <typename T>
    void assemble_cells(
        const std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                                const std::int32_t*, const T*)>& mat_set,
        const dolfinx::mesh::Mesh& cell_mesh,
        const dolfinx::mesh::Mesh& facet_mesh,
        const xtl::span<const std::int32_t>& cells,
        const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap0, int bs0,
        const dolfinx::graph::AdjacencyList<std::int32_t>& dofmap1, int bs1,
        const std::vector<bool>& bc0, const std::vector<bool>& bc1,
        const std::function<void(T*, const T*, const T*, const double*, const int*,
                                const std::uint8_t*)>& kernel,
        const xtl::span<const T>& coeffs, int cstride,
        const xtl::span<const T>& constants,
        const std::function<std::uint8_t(std::size_t)>& get_perm)
    {
        const int tdim = cell_mesh.topology().dim();

        // Prepare cell geometry
        const dolfinx::graph::AdjacencyList<std::int32_t>& x_dofmap =
            cell_mesh.geometry().dofmap();

        // FIXME: Add proper interface for num coordinate dofs
        const std::size_t num_dofs_g = x_dofmap.num_links(0);
        const xt::xtensor<double, 2>& x_g = cell_mesh.geometry().x();

        // FIXME: Find better way to handle facet geom
        const std::size_t facet_num_dofs_g =
            facet_mesh.geometry().dofmap().num_links(0);

        const int num_cell_facets
            = dolfinx::mesh::cell_num_entities(cell_mesh.topology().cell_type(), tdim - 1);
        
        // Data structures used in assembly
        std::vector<double> coordinate_dofs(3 * (num_dofs_g + num_cell_facets * facet_num_dofs_g));
        const int num_dofs0 = dofmap0.links(0).size();
        const int num_dofs1 = dofmap1.links(0).size();
        const int ndim0 = bs0 * num_dofs0;
        const int ndim1 = bs1 * num_dofs1;
        std::vector<std::uint8_t> cell_facet_perms(num_cell_facets);
        auto c_to_f = cell_mesh.topology().connectivity(tdim, tdim - 1);

        xt::xarray<T> Ae_sc = xt::zeros<T>({ndim0 * num_cell_facets,
                                            ndim1 * num_cell_facets});
        
        for (auto cell : cells)
        {
            auto cell_facets = c_to_f->links(cell);
            auto ent_to_geom = dolfinx::mesh::entities_to_geometry(
                cell_mesh, tdim - 1, cell_facets, false);
	
            dolfinx_hdg::fem::impl_helpers::get_coordinate_dofs(
                coordinate_dofs, cell, cell_facets, x_dofmap, x_g, ent_to_geom);
            
            dolfinx_hdg::fem::impl_helpers::get_cell_facet_perms(
                cell_facet_perms, cell, num_cell_facets, get_perm);
            
            std::cout << "Cell " << cell << "\n";
            for (int i = 0; i < num_cell_facets; ++i)
            {
                std::cout << "  facet = " << cell_facets[i] << " perm = " << unsigned(cell_facet_perms[i]) << "\n";
            }

            std::fill(Ae_sc.begin(), Ae_sc.end(), 0);
            kernel(Ae_sc.data(), coeffs.data() + cell * cstride, constants.data(),
                   coordinate_dofs.data(), nullptr, cell_facet_perms.data());

            std::cout << "Ae_sc\n" << Ae_sc << "\n";

            // Double loop over cell facets to assemble
            // FIXME Is there a better way?
            for (int local_f_i = 0; local_f_i < num_cell_facets; ++local_f_i)
            {
                for (int local_f_j = 0; local_f_j < num_cell_facets; ++local_f_j)
                {
                    const int f_i = cell_facets[local_f_i];
                    const int f_j = cell_facets[local_f_j];

                    auto dofs0 = dofmap0.links(f_i);
                    auto dofs1 = dofmap1.links(f_j);

                    // Matrix corresponding to dofs of facets f_i and f_j
                    // NOTE Have to cast to xt::xarray<double> (can't just use auto)
                    // otherwise it returns a view and Ae_sc_f_ij.data() gets the
                    // wrong values (the values) from the full Ae_sc array
                    xt::xarray<double> Ae_sc_f_ij =
                        xt::view(Ae_sc,
                                 xt::range(local_f_i * ndim0,
                                           local_f_i * ndim0 + ndim0),
                                 xt::range(local_f_j * ndim1,
                                           local_f_j * ndim1 + ndim1));

                    // FIXME Might be better/more efficient to do this to Ae_sc
                    // directly.
                    if (!bc0.empty())
                    {
                        for (int i = 0; i < num_dofs0; ++i)
                        {
                            for (int k = 0; k < bs0; ++k)
                            {
                                if (bc0[bs0 * dofs0[i] + k])
                                {
                                    // Zero row bs0 * i + k
                                    const int row = bs0 * i + k;
                                    xt::view(Ae_sc_f_ij, row, xt::all()) = 0.0;
                                }
                            }
                        }
                    }

                    if (!bc1.empty())
                    {
                        for (int j = 0; j < num_dofs1; ++j)
                        {
                            for (int k = 0; k < bs1; ++k)
                            {
                                if (bc1[bs1 * dofs1[j] + k])
                                {
                                    // Zero column bs1 * j + k
                                    const int col = bs1 * j + k;
                                    xt::view(Ae_sc_f_ij, xt::all(), col) = 0.0;
                                }
                            }
                        }
                    }

                    mat_set(dofs0.size(), dofs0.data(),
                            dofs1.size(), dofs1.data(),
                            Ae_sc_f_ij.data());
                }
            }
        }
    }

    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_set,
        const dolfinx::fem::Form<T> &a,
        const xtl::span<const T> &constants,
        const xtl::span<const T> &coeffs,
        int cstride,
        const std::vector<bool> &bc0,
        const std::vector<bool> &bc1)
    {
        // FIXME Vector elements (i.e. block size \neq 1) might break some of this
        std::shared_ptr<const dolfinx::mesh::Mesh> cell_mesh = a.mesh();
        assert(cell_mesh);

        std::shared_ptr<const dolfinx::mesh::Mesh> facet_mesh =
            a.function_spaces().at(0)->mesh();
        assert(facet_mesh);

        // Get dofmap data
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap0
            = a.function_spaces().at(0)->dofmap();
        std::shared_ptr<const dolfinx::fem::DofMap> dofmap1
            = a.function_spaces().at(1)->dofmap();
        assert(dofmap0);
        assert(dofmap1);
        const dolfinx::graph::AdjacencyList<std::int32_t>& dofs0 = dofmap0->list();
        const int bs0 = dofmap0->bs();
        const dolfinx::graph::AdjacencyList<std::int32_t>& dofs1 = dofmap1->list();
        const int bs1 = dofmap1->bs();

        // TODO dof transformations and facet permutations
        std::function<std::uint8_t(std::size_t)> get_perm;
        if (a.needs_facet_permutations())
        {
            std::cout << "Assemble matrix perms\n";
            cell_mesh->topology_mutable().create_entity_permutations();
            const std::vector<std::uint8_t>& perms
                = cell_mesh->topology().get_facet_permutations();
            get_perm = [&perms](std::size_t i) { return perms[i]; };
        }
        else
            get_perm = [](std::size_t) { return 0; };

        for (int i : a.integral_ids(dolfinx::fem::IntegralType::cell))
        {
            const auto& fn = a.kernel(dolfinx::fem::IntegralType::cell, i);
            const std::vector<std::int32_t>& cells = a.cell_domains(i);
            const int tdim = cell_mesh->topology().dim();
            cell_mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
            impl::assemble_cells<T>(mat_set, *cell_mesh, *facet_mesh, cells, dofs0, bs0,
                                    dofs1, bs1, bc0, bc1, fn, coeffs,
                                    cstride, constants, get_perm);
        }
    }
}
