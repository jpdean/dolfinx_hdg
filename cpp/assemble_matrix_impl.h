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
#include <dolfinx/common/array2d.h>
#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp> // To cout xtensor arrays

namespace dolfinx_hdg::fem::impl
{
    template <typename T>
    void assemble_matrix(
        const std::function<int(std::int32_t, const std::int32_t *, std::int32_t,
                                const std::int32_t *, const T *)> &mat_set,
        const std::vector<std::vector<std::shared_ptr<
            const dolfinx::fem::Form<T>>>> &a,
        const xtl::span<const T> &constants,
        const dolfinx::array2d<T> &coeffs, const std::vector<bool> &bc0,
        const std::vector<bool> &bc1)
    {
        std::shared_ptr<const dolfinx::mesh::Mesh> mesh = a[0][0]->mesh();
        assert(mesh);
        const int tdim = mesh->topology().dim();

        const dolfinx::graph::AdjacencyList<std::int32_t> &x_dofmap =
            mesh->geometry().dofmap();
        const std::size_t num_dofs_g = x_dofmap.num_links(0);
        const xt::xtensor<double, 2> &x_g = mesh->geometry().x();
        std::vector<double> coordinate_dofs(3 * num_dofs_g);

        // TODO Get dofmap data
        // TODO Dof transformation stuff

        // FIXME Think of how to do this nicely. Should use integral_ids
        // FIXME TODO Permutations. See dolfinx exterior facet integrals
        // TODO Active cells. NOTE For the facet space, getting the active
        // cells in the usual way below actually gets the active facet
        // numbers, as the facets are treated as cells

        // FIXME What if this has no cell or facet kernel etc.?
        const auto &a00_cell_kernel =
            a[0][0]->kernel(dolfinx::fem::IntegralType::cell, -1);
        const auto &a00_facet_kernel =
            a[0][0]->kernel(dolfinx::fem::IntegralType::exterior_facet, -1);

        // Dofmaps and block size for the a00 form
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap00_0 =
            a[0][0]->function_spaces().at(0)->dofmap()->list();
        const dolfinx::graph::AdjacencyList<std::int32_t> &dofmap00_1 =
            a[0][0]->function_spaces().at(1)->dofmap()->list();
        const int bs00_0 = a[0][0]->function_spaces().at(0)->dofmap()->bs();
        const int bs00_1 = a[0][0]->function_spaces().at(1)->dofmap()->bs();
        const int num_dofs00_0 = bs00_0 * dofmap00_0.links(0).size();
        const int num_dofs00_1 = bs00_1 * dofmap00_1.links(0).size();

        // TODO Is this needed?
        mesh->topology_mutable().create_connectivity(tdim, tdim - 1);
        auto c_to_f = mesh->topology().connectivity(tdim, tdim - 1);
        assert(c_to_f);

        for (int c = 0; c < c_to_f->num_nodes(); ++c)
        {
            xt::xarray<double> Ae00 = xt::zeros<double>({num_dofs00_0,
                                                         num_dofs00_1});

            // Get cell coordinates/geometry
            auto x_dofs = x_dofmap.links(c);
            for (std::size_t i = 0; i < x_dofs.size(); ++i)
            {
                std::copy_n(xt::row(x_g, x_dofs[i]).begin(), 3,
                            std::next(coordinate_dofs.begin(), 3 * i));
            }

            // FIXME Passing cell to coeffs that were created for facets.
            // FIXME Check that Ae00.data() for an xtensor array is correct here
            // NOTE These FIXMEs apply to other kernel calls too
            a00_cell_kernel(Ae00.data(), coeffs.row(c).data(), constants.data(),
                            coordinate_dofs.data(), nullptr, nullptr);

            for (auto f : c_to_f->links(c))
            {
                // Get local index of facet with respect to the cell
                // TODO Could just loop through local facets or do this
                // some other way
                auto facets = c_to_f->links(c);
                auto it = std::find(facets.begin(), facets.end(), f);
                assert(it != facets.end());
                const int local_f = std::distance(facets.begin(), it);

                std::cout << "f = " << f << "  local_f = " << local_f << "\n";

                // TODO Permutatations
                a00_cell_kernel(Ae00.data(), coeffs.row(c).data(), constants.data(),
                                coordinate_dofs.data(), &local_f, nullptr);
            }

            std::cout << c << "\n";
            std::cout << Ae00 << "\n";
        }
    }
}
